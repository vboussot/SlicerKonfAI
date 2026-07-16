# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""App selection panel: app list population (Hugging Face, remote server,
local folders), app management actions and description display."""

import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import slicer
from qt import (
    QCheckBox,
    QCursor,
    QFileDialog,
    QIcon,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QSettings,
    QSize,
)
from slicer.i18n import tr as _

from KonfAILib.logic.hf import get_hf_app_file_list
from KonfAILib.platform_utils import open_path_in_file_browser
from KonfAILib.widgets.dialogs.download_files import DownloadFilesDialog
from KonfAILib.widgets.helpers import resource_path
from KonfAILib.widgets.panels.base import KonfAIAppPanel

if TYPE_CHECKING:
    from KonfAILib.widgets.app_template import KonfAIAppTemplateWidget


class KonfAIAppSelectionPanel(KonfAIAppPanel):
    """
    App selection panel of a KonfAI application.

    This panel owns the app combo box and its management buttons
    (add/remove/config/refresh), the app description display and the
    list of locally registered app repositories.
    """

    def __init__(self, template: "KonfAIAppTemplateWidget") -> None:
        super().__init__(template, "KonfAIAppSelection")

        self._description_expanded = False

        # App selection and management
        self.ui.addAppButton.clicked.connect(self.on_add_app)
        self.ui.removeAppButton.clicked.connect(self.on_remove_app)

        # Configuration button (opens folder containing KonfAI YAML configs)
        self.ui.configButton.clicked.connect(self.on_open_config)

        self.ui.refreshAppsListButton.setIcon(QIcon(resource_path("Icons/refresh.png")))
        self.ui.refreshAppsListButton.setIconSize(QSize(18, 18))
        self.ui.refreshAppsListButton.clicked.connect(self.on_refresh_app)

        # Description toggle and app selection changes
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.appComboBox.currentIndexChanged.connect(self.on_app_selected)

        self.app_local_repositoy: list[str] = []

    def on_refresh_app(self):
        from konfai_apps.app_repository import AppRepositoryError

        try:
            self.populate_apps(True)
        except AppRepositoryError as e:
            slicer.util.errorDisplay(
                "Unable to refresh the list of applications.\n\n"
                "This may happen if you are offline, if the repository is not accessible, "
                "The application list has not been updated.",
                detailedText=getattr(e, "details", None) or str(e),
            )

    @contextmanager
    def _apps_loading_feedback(self, initial_message: str):
        controls = [
            self.ui.appComboBox,
            self.ui.addAppButton,
            self.ui.removeAppButton,
            self.ui.configButton,
            self.ui.refreshAppsListButton,
        ]
        previous_states = [(control, control.enabled) for control in controls]
        for control, _prev in previous_states:
            control.enabled = False

        with self.transient_status_progress(initial_message):
            try:
                yield
            finally:
                for control, was_enabled in previous_states:
                    control.enabled = was_enabled

    def populate_apps(self, force_update: bool = False) -> None:
        remote_server, ok = self.get_remote_server()
        from konfai.utils.errors import AppRepositoryError
        from konfai_apps.app_repository import (
            get_app_repository_info,
            get_available_apps_on_hf_repo,
            get_available_apps_on_remote_server,
        )

        with self._apps_loading_feedback("Loading applications..."):
            apps_name: list[str] = []
            if remote_server is None:
                settings = QSettings()
                raw = settings.value(f"KonfAI-Settings/{self._name}/Apps")
                apps_name = []
                if raw is not None:
                    apps_name = json.loads(raw)
                default_apps_name = []

                repo_count = max(len(self.template._konfai_repo_list), 1)
                for idx, konfai_repo in enumerate(self.template._konfai_repo_list, start=1):
                    self._set_status_progress(
                        int(((idx - 1) / repo_count) * 40),
                        f"Checking {konfai_repo} on Hugging Face...",
                    )
                    try:
                        default_apps_name += [
                            konfai_repo + ":" + app_name
                            for app_name in get_available_apps_on_hf_repo(konfai_repo, force_update)
                        ]
                    except AppRepositoryError as e:
                        slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
                        return
                    self._set_status_progress(
                        int((idx / repo_count) * 40),
                        f"Loaded app names from {konfai_repo}.",
                    )
                apps_name = list(set(default_apps_name + apps_name))
                self.app_local_repositoy = apps_name
            elif ok:
                self._set_status_progress(10, "Loading applications from remote server...")
                apps_name = [
                    f"{remote_server.host}:{remote_server.port}:{app_name}|{remote_server.token}"
                    for app_name in get_available_apps_on_remote_server(remote_server)
                ]
                self._set_status_progress(40, "Loaded application names from remote server.")

            # Populate the app combo box with apps found in the provided
            # Hugging Face repos or available app on remote server
            def _is_missing_app_error(exc: Exception) -> bool:
                """True iff the app is genuinely gone (HTTP 404 / entry-not-found), NOT a transient
                network / offline failure. Only a genuinely-missing app is pruned from the saved list,
                so an app you own does not vanish just because you happened to be offline."""
                seen: set[int] = set()
                cur: BaseException | None = exc
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    if type(cur).__name__ in ("EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError"):
                        return True
                    if getattr(getattr(cur, "response", None), "status_code", None) == 404:
                        return True
                    cur = cur.__cause__ or cur.__context__
                text = str(exc).lower()
                return any(marker in text for marker in ("404", "entry not found", "does not exist", "not found"))

            apps = []
            removed_missing: list[str] = []
            sorted_apps_name = sorted(apps_name)
            if not sorted_apps_name:
                self._set_status_progress(100, "No applications found.")
            for idx, app_name in enumerate(sorted_apps_name, start=1):
                self._set_status_progress(
                    40 + int((idx / len(sorted_apps_name)) * 60),
                    f"Initializing {app_name}...",
                )
                try:
                    apps.append(get_app_repository_info(app_name, False))
                except Exception as e:  # noqa: BLE001 - one bad app must not abort the whole list
                    if _is_missing_app_error(e):
                        # Deleted / renamed on its source (404): drop it so it stops being re-fetched.
                        removed_missing.append(app_name)
                        self._update_logs(f"Removed '{app_name}': no longer available on its source (404).", False)
                    else:
                        # Transient (offline / server down): keep it; it will resolve next time online.
                        self._update_logs(f"Skipped '{app_name}': temporarily unreachable ({e}).", False)
            if removed_missing:
                self.app_local_repositoy = [n for n in self.app_local_repositoy if n not in removed_missing]
                QSettings().setValue(
                    f"KonfAI-Settings/{self._name}/Apps",
                    json.dumps(self.app_local_repositoy),
                )

            self._set_status_progress(100, f"Loaded {len(apps)} applications.")

        was_blocked = self.ui.appComboBox.blockSignals(True)
        self.ui.appComboBox.clear()
        for app in apps:
            self.ui.appComboBox.addItem(app.get_display_name(), app)
        app_param = self.get_parameter("App")
        index = 0
        for i in range(self.ui.appComboBox.count):
            app = self.ui.appComboBox.itemData(i)
            if app.get_name() == app_param:
                index = i
                break
        self.ui.appComboBox.setCurrentIndex(index)
        self.ui.appComboBox.blockSignals(was_blocked)
        self.on_app_selected()

    def on_app_selected(self) -> None:
        """
        Handle app selection changes.

        This method:
          - Updates the app management buttons and the description display
          - Stores the selected app in the parameter node
          - Notifies the template so app-dependent widgets are reconfigured
        """
        app = self.ui.appComboBox.currentData
        if app is None:
            return

        # Removing app from disk is only relevant for custom (local) apps;
        # by default we disable the button for HF apps.
        remote_server, _ = self.get_remote_server()
        if remote_server is not None:
            self.ui.removeAppButton.setEnabled(False)
            self.ui.addAppButton.setEnabled(False)
        else:
            self.ui.removeAppButton.setEnabled(app.get_name().split(":")[0] not in self.template._konfai_repo_list)
            self.ui.addAppButton.setEnabled(True)
            self.ui.configButton.setEnabled(True)

        from konfai_apps.app_repository import LocalAppRepositoryFromDirectory

        if isinstance(app, LocalAppRepositoryFromDirectory):
            self.ui.configButton.setIcon(QIcon(resource_path("Icons/gear.png")))
            self.ui.configButton.setIconSize(QSize(18, 18))
            self.ui.configButton.toolTip = "Open KonfAi app config folder"
        else:
            self.ui.configButton.setIcon(QIcon(resource_path("Icons/download.png")))
            self.ui.configButton.setIconSize(QSize(18, 18))
            self.ui.configButton.toolTip = "Download KonfAi app"
        # Reset description expansion state and update description label
        self._description_expanded = False
        self.on_toggle_description()

        self.set_parameter("App", app.get_name())
        self.template.dispatch_app_changed(app)

    def _refresh_selected_app_metadata(self):
        """
        Reload the selected app metadata from its repository source.

        This keeps checkpoint availability in sync after commands that may
        download or update model files as a side effect.
        """
        app = self.ui.appComboBox.currentData
        index = self.ui.appComboBox.currentIndex
        if app is None or index < 0:
            return None

        from konfai_apps.app_repository import get_app_repository_info

        try:
            refreshed_app = get_app_repository_info(app.get_name(), False)
        except Exception as exc:
            self._update_logs(f"Unable to refresh app metadata for {app.get_display_name()}: {exc}", False)
            return None

        was_blocked = self.ui.appComboBox.blockSignals(True)
        self.ui.appComboBox.setItemData(index, refreshed_app)
        self.ui.appComboBox.blockSignals(was_blocked)
        self.on_app_selected()
        return refreshed_app

    def on_toggle_description(self) -> None:
        """
        Toggle between short and full app description text in the UI.
        """
        app = self.ui.appComboBox.currentData
        if not app:
            return

        if self._description_expanded:
            self.ui.appDescriptionLabel.setText(app.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.appDescriptionLabel.setText(app.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")

        self._description_expanded = not self._description_expanded

    def on_remove_app(self) -> None:
        """
        Remove the currently selected app from the list, optionally deleting
        the corresponding directory from disk.

        This is intended for locally added apps, not Hugging Face apps.
        """
        from konfai_apps.app_repository import LocalAppRepositoryFromDirectory

        app = self.ui.appComboBox.currentData
        index = self.ui.appComboBox.currentIndex
        if index < 0:
            return
        # A broken/unresolvable entry (e.g. its source 404'd) has no app object; fall back to the
        # displayed label so a dead entry can still be removed instead of crashing on ``None``.
        label = app.get_display_name() if app is not None else (self.ui.appComboBox.currentText or "this app")

        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("Remove app?")
        mb.setText(f"Do you really want to remove “{label}” from the list?")
        mb.setInformativeText("This will remove the app entry from the extension’s list.")
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)

        chk = QCheckBox("Also delete the folder from disk")
        chk.setChecked(False)
        mb.setCheckBox(chk)

        if mb.exec_() != QMessageBox.Yes:
            return

        self.ui.appComboBox.removeItem(index)
        name = app.get_name() if app is not None else None
        if name is not None and name in self.app_local_repositoy:
            self.app_local_repositoy.remove(name)
        QSettings().setValue(
            f"KonfAI-Settings/{self._name}/Apps",
            json.dumps(self.app_local_repositoy),
        )

        if chk.isChecked():
            if isinstance(app, LocalAppRepositoryFromDirectory):
                try:
                    shutil.rmtree(app.get_name())
                    QMessageBox.information(
                        None, "Folder deleted", f"The folder has been successfully deleted:\n{app.get_name()}"
                    )
                except Exception as e:
                    QMessageBox.critical(None, "Deletion error", f"Failed to delete folder:\n{app.get_name()}\n\n{e}")

    def on_open_config(self) -> None:
        """
        Open the directory containing the inference configuration

        The folder is opened in the operating system's default file browser.
        """
        app = self.ui.appComboBox.currentData
        if app is None:
            return
        from konfai_apps.app_repository import LocalAppRepositoryFromHF

        if isinstance(app, LocalAppRepositoryFromHF):
            try:
                all_files = get_hf_app_file_list(app._repo_id, app._app_name)
            except Exception as exc:
                QMessageBox.critical(None, "Hugging Face", str(exc))
                return

            def open_local_app_folder() -> None:
                try:
                    config_files = app.download_config_file()
                except Exception as exc:
                    QMessageBox.critical(None, "Hugging Face", f"Unable to access the local app folder.\n\n{exc}")
                    return

                if not config_files:
                    QMessageBox.warning(None, "Hugging Face", "No local app folder is available yet.")
                    return

                folder = config_files[0].parent
                opened, error = open_path_in_file_browser(folder)
                if not opened:
                    QMessageBox.critical(
                        None,
                        "Open folder",
                        f"Could not open folder:\n{folder}",
                        detailedText=error or "",
                    )

            dlg = DownloadFilesDialog(
                all_files,
                app.get_checkpoints_name_available(),
                on_open_folder=open_local_app_folder,
            )
            if dlg.exec() != dlg.Accepted:
                return
            selected_files = dlg.selected_files()
            if not selected_files:
                return

            import sys
            import textwrap

            pycode = textwrap.dedent(f"""
                from konfai_apps.app_repository import LocalAppRepositoryFromHF
                from konfai.utils.runtime import MinimalLog

                with MinimalLog() as log:
                    selected_files = {selected_files!r}
                    for filename in selected_files:
                        LocalAppRepositoryFromHF.download(
                            "{app._repo_id}",
                            "{app._app_name}" + "/" + filename,
                            True,
                        )
                        print(f"[KonfAI-Apps] {{filename}} is ready.")
                """)

            def on_end_function() -> None:
                self._refresh_selected_app_metadata()

            self._update_logs("Starting Hugging Face download...", True)
            self._update_progress(0, "")
            self.set_running(True)
            self.template.ui.runInferenceButton.enabled = True
            self.template.ui.runInferenceButton.toolTip = _("Stop download")
            self.process.run(sys.executable, Path("./").resolve(), ["-c", pycode], on_end_function)
        else:
            config_files = app.download_config_file()
            opened, error = open_path_in_file_browser(config_files[0].parent)
            if not opened:
                slicer.util.errorDisplay(f"Could not open folder:\n{config_files[0].parent}", detailedText=error or "")

    def on_add_app(self) -> None:
        """
        Show a menu to add a new app (from a folder, from Hugging Face, or configure fine-tuning).
        """
        m = QMenu()
        act_folder = m.addAction("Add from folder…")
        act_hf = m.addAction("Add from Hugging Face…")
        act_ft = m.addAction("Setup fine-tuning")
        can_setup_fine_tuning = self._has_root_training_config(self.ui.appComboBox.currentData)
        act_ft.setEnabled(can_setup_fine_tuning)
        if not can_setup_fine_tuning:
            act_ft.setToolTip("Fine-tuning setup requires a root Config.yml file in the selected app.")
        chosen = m.exec_(QCursor.pos())
        if chosen is None:
            return
        if chosen is act_folder:
            self.on_add_folder()
        elif chosen is act_hf:
            self.on_add_hf()
        elif chosen is act_ft:
            self.on_add_ft()

    def on_add_folder(self) -> None:
        """
        Add a app located in a local folder to the apps list.

        The folder is expected to contain a valid KonfAI app configuration (YAML + weights).
        """
        from konfai_apps.app_repository import LocalAppRepositoryFromDirectory

        app_dir = QFileDialog.getExistingDirectory(None, "Select App Folder", os.path.expanduser("~"))
        if not app_dir:
            return
        try:
            app = LocalAppRepositoryFromDirectory(Path(app_dir).parent, Path(app_dir).name)
        except Exception as e:
            slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
            return

        # Ensure uniqueness of display names to avoid confusion in the combo box
        items = [self.ui.appComboBox.itemText(i) for i in range(self.ui.appComboBox.count)]
        if app.get_display_name() in items:
            QMessageBox.critical(
                None,
                "App already listed",
                f'The app "{app.get_display_name()}" is already in the list.',
            )
            return
        self.ui.appComboBox.addItem(app.get_display_name(), app)
        self.ui.appComboBox.setCurrentIndex(self.ui.appComboBox.findData(app))
        self.app_local_repositoy.append(app.get_name())

    def ask_subdir(self, dirs: list[str]) -> str | None:
        """
        Ask the user to select a subdirectory from a list of directory names.

        Parameters
        ----------
        dirs : list[str]
            Available subdirectories to choose from.

        Returns
        -------
        str | None
            The selected directory name or None if the user cancels.
        """
        dlg = QInputDialog()
        dlg.setWindowTitle("Select subdirectory")
        dlg.setLabelText("Choose a directory:")
        dlg.setComboBoxItems(dirs)
        dlg.setOption(QInputDialog.UseListViewForComboBoxItems)

        if dlg.exec_() != dlg.Accepted:
            return None

        return dlg.textValue()

    def _has_root_training_config(self, app) -> bool:
        """
        Return whether the selected app exposes a root-level ``Config.yml`` file.

        Fine-tuning setup currently expects the training configuration to be
        copied to ``./Config.yml`` in the generated app folder, so nested config
        files are not treated as valid for this action.
        """
        if app is None:
            return False

        from konfai_apps.app_repository import (
            AppRepositoryInfoFromRemoteServer,
            LocalAppRepositoryFromDirectory,
            LocalAppRepositoryFromHF,
        )

        try:
            if isinstance(app, LocalAppRepositoryFromDirectory):
                filenames = LocalAppRepositoryFromDirectory.get_filenames(app._app_directory, app._app_name)
            elif isinstance(app, LocalAppRepositoryFromHF):
                filenames = LocalAppRepositoryFromHF.get_filenames(app._repo_id, app._app_name, False)
            elif isinstance(app, AppRepositoryInfoFromRemoteServer):
                return False
            else:
                return False
        except Exception:
            return False

        return "Config.yml" in filenames

    def on_add_hf(self) -> None:
        """
        Add a app from a Hugging Face repository.

        The user is prompted for a repo id (e.g., 'VBoussot/ImpactSynth') and then
        a subdirectory (app name) is selected among the available directories.
        """
        from huggingface_hub import HfApi
        from konfai_apps.app_repository import LocalAppRepositoryFromHF

        text = QInputDialog().getText(
            self,
            "Add from Hugging Face",
            "Enter repo id (e.g. VBoussot/ImpactSynth):",
            QLineEdit.Normal,
        )
        repo_id = text.strip()
        if not repo_id:
            return

        api = HfApi()
        base_repo_id, _, revision = repo_id.partition("@")

        files = None
        try:
            with self._apps_loading_feedback(f"Checking {repo_id} on Hugging Face..."):
                self._set_status_progress(20, f"Connecting to {repo_id}...")
                files = api.list_repo_files(
                    repo_id=base_repo_id,
                    revision=revision or None,
                    repo_type="model",
                )
                self._set_status_progress(100, f"Loaded repository tree for {repo_id}.")
        except Exception:
            QMessageBox.critical(None, "Hugging Face", f"Repository '{repo_id}' does not exist on Hugging Face.")
            return

        # List top-level directories as candidate app names
        dirs = sorted({path.split("/")[0] for path in files if "/" in path})
        app_name = self.ask_subdir(dirs)
        if app_name:
            from konfai_apps.app_repository import is_app_repo

            app_files = [path.removeprefix(f"{app_name}/") for path in files if path.startswith(f"{app_name}/")]
            state = is_app_repo(app_files)
            if not state:
                QMessageBox.critical(
                    None,
                    "App",
                    f"The selected folder '{app_name}' does not contain a valid KonfAI app structure.",
                )
                return

            with self._apps_loading_feedback(f"Initializing {app_name}..."):
                self._set_status_progress(60, f"Loading metadata for {app_name}...")
                app = LocalAppRepositoryFromHF(repo_id, app_name, True)
                self._set_status_progress(100, f"Loaded {app_name}.")

            # Do not add duplicate display names
            items = [self.ui.appComboBox.itemText(i) for i in range(self.ui.appComboBox.count)]
            if app.get_display_name() in items:
                QMessageBox.critical(
                    None,
                    "App already listed",
                    f'The app "{app.get_display_name()}" is already in the list.',
                )
                return

            self.ui.appComboBox.addItem(app.get_display_name(), app)
            self.ui.appComboBox.setCurrentIndex(self.ui.appComboBox.findData(app))
            self.app_local_repositoy.append(app.get_name())

    def on_add_ft(self) -> None:
        """
        Create a new folder intended for fine-tuning a app and add it to the list.

        This helper guides the user through selecting a parent directory,
        naming the fine-tune app folder and creating a basic skeleton with a README.
        """
        from konfai_apps.app_repository import LocalAppRepositoryFromDirectory

        app = self.ui.appComboBox.currentData
        if app is None:
            return
        if not self._has_root_training_config(app):
            QMessageBox.warning(
                None,
                "Fine tune",
                "The selected app does not contain a root Config.yml file required for fine-tuning setup.",
            )
            return

        # Choose the parent directory for the new app
        parent_dir = QFileDialog.getExistingDirectory(None, "Choose parent directory for the new app")
        if not parent_dir:
            return

        # Ask for the new app name
        name = QInputDialog.getText(None, "Fine tune", "New app name (folder will be created):")
        if not name.strip():
            return

        # Ask for the new app diplay name
        display_name = QInputDialog.getText(None, "Fine tune", "New diplay app name:")
        if not display_name.strip():
            return

        items = [self.ui.appComboBox.itemText(i) for i in range(self.ui.appComboBox.count)]
        if display_name in items:
            QMessageBox.critical(
                None,
                "App already listed",
                f'The app "{display_name}" is already in the list.',
            )
            return

        # Ask number of epochs
        epochs = QInputDialog.getInt(None, "Fine tune", "Number of epochs:", 10, 1, 10000, 1)

        it_validation = QInputDialog.getInt(
            None, "Fine tune", "Validation interval (every N epochs):", 1000, 1, 100000, 1
        )
        # Make a filesystem-safe name (slug)
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        if not safe:
            QMessageBox.warning(None, "Invalid name", "Please enter a valid name.")
            return

        # [] selects the checkpoint(s) to fine-tune: empty means the default advertised in app.json.
        app.install_fine_tune("Config.yml", Path(parent_dir) / name, display_name, epochs, it_validation, [])
        # Add the folder to the app combo box
        app_ft = LocalAppRepositoryFromDirectory(Path(parent_dir), name)
        self.ui.appComboBox.addItem(app_ft.get_display_name(), app_ft)
        self.ui.appComboBox.setCurrentIndex(self.ui.appComboBox.findData(app_ft))
        self.app_local_repositoy.append(app_ft.get_name())

    def cleanup(self) -> None:
        """
        Persist the locally registered apps when the module is closed.
        """
        settings = QSettings()
        if len(self.app_local_repositoy) > 0:
            settings.setValue(
                f"KonfAI-Settings/{self._name}/Apps",
                json.dumps(self.app_local_repositoy),
            )
