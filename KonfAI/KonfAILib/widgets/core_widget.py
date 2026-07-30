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

"""Core widget shared by KonfAI and the sister extensions: remote servers,
device selection, RAM/VRAM monitoring, logs/progress, app registration."""

import itertools
import json
import os
from typing import Any

import slicer
import vtk
from qt import (
    QCursor,
    QDialog,
    QIcon,
    QSettings,
    QMenu,
    QSize,
    Qt,
    QTabWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer.util import VTKObservationMixin

from KonfAILib.logic.dependencies import install_konfai, install_torch
from KonfAILib.logic.servers import RemoteServer
from KonfAILib.platform_utils import open_path_in_file_browser
from KonfAILib.widgets.app_template import AppTemplateWidget
from KonfAILib.widgets.dialogs.remote_server import RemoteServerAddDialog, RemoteServerConfigDialog
from KonfAILib.widgets.helpers import is_dark_theme, resource_path, slicer_wait_popup


def _resource_bar_qss(usage: float) -> str:
    """Flat rounded gauge for the thin RAM/VRAM bars; the chunk turns red above 80% usage."""
    color = "#e74c3c" if usage > 0.8 else "#2ecc71"
    return (
        "QProgressBar { border: none; background: rgba(128,128,128,0.18); border-radius: 3px; }"
        f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"
    )


class KonfAICoreWidget(QWidget, VTKObservationMixin, ScriptedLoadableModuleLogic):
    """
    Core KonfAI widget responsible for:

      - Global device selection (CPU/GPU and VRAM monitoring)
      - Log and progress display
      - RAM monitoring
      - Management of KonfAI applications (tabs)
      - Persistence of global parameters via a parameter node
    """

    def __init__(self, title: str) -> None:
        """
        Initialize the core KonfAI widget.

        Parameters
        ----------
        title : str
            Title displayed in the header of the KonfAI module.
        """
        QWidget.__init__(self)
        VTKObservationMixin.__init__(self)
        ScriptedLoadableModuleLogic.__init__(self)

        self._parameter_node = None
        self._updatingGUIFromParameterNode = False
        self._apps: dict[str, AppTemplateWidget] = {}
        self._current_konfai_app: AppTemplateWidget | None = None

        # Load the main KonfAICore UI from .ui file
        ui_widget = slicer.util.loadUI(resource_path("UI/KonfAICore.ui"))
        self.setLayout(QVBoxLayout())
        # The .ui root layout already carries the margins; a second layer here doubles every gap
        # (header-to-content in particular).
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.initialize_parameter_node()
        # Quiet header, one language for every extension: the title in spaced small caps between
        # the two native thin rules, with the Studio launcher closing the row.
        self.ui.headerLogoLabel.hide()
        self.ui.headerTitleLabel.text = title.upper()
        self.ui.headerTitleLabel.setStyleSheet(
            "background: transparent; font-weight: 700; font-size: 12px; letter-spacing: 2px;"
            " color: rgba(128,144,160,0.95); padding: 0 6px;"
        )

        # Console-style log area, matched to the theme. A full widget stylesheet is required either
        # way: font-only styling makes the widget drop the theme palette and render white in dark mode.
        log_bg, log_fg = ("#1f2123", "#d8dadc") if is_dark_theme() else ("#fbfcfd", "#2b3440")
        self.ui.logText.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {log_bg}; color: {log_fg}; font-family: monospace;"
            " font-size: 11px; border: 1px solid rgba(128,128,128,0.35); border-radius: 4px; }"
        )

        # Observe scene close/open to keep parameter node in sync
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.on_scene_start_close)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.on_scene_end_close)

        # Work directory button configuration
        self.ui.openTempButton.setIcon(QIcon(resource_path("Icons/folder.png")))
        self.ui.openTempButton.setIconSize(QSize(18, 18))
        self.ui.openTempButton.clicked.connect(self.on_open_work_dir)
        self.ui.openTempButton.setEnabled(False)

        # Remote Server Configuration button
        self.ui.remoteServerConfigButton.setIcon(QIcon(resource_path("Icons/gear.png")))
        self.ui.remoteServerConfigButton.setIconSize(QSize(18, 18))
        self.ui.remoteServerConfigButton.clicked.connect(self.on_remote_server_config)

        self.ui.remoteServerAddButton.clicked.connect(self.on_add_remote_server)
        self.ui.remoteServerComboBox.currentIndexChanged.connect(self.on_remote_server_changed)

        # Studio launcher: a small flat robot icon button, same language as the module's other icon
        # buttons. A right-click menu offers to stop the detached server (no terminal to Ctrl+C).
        self.ui.studioButton.setIcon(QIcon(resource_path("Icons/studio.png")))
        self.ui.studioButton.setIconSize(QSize(22, 22))
        self.ui.studioButton.setAutoRaise(True)
        self.ui.studioButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.ui.studioButton.clicked.connect(self.on_open_studio)
        studio_menu = QMenu(self.ui.studioButton)
        studio_menu.addAction("Stop the Studio server", self.on_stop_studio)
        studio_menu.addAction("Choose Studio executable…", self.on_choose_studio_executable)
        self.ui.studioButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.studioButton.customContextMenuRequested.connect(
            lambda pos: studio_menu.exec_(self.ui.studioButton.mapToGlobal(pos))
        )

        self.ui.deviceComboBox.currentIndexChanged.connect(self.on_device_changed)

        # Idle refresh of the RAM/VRAM gauges: log lines only update them during a run, so a timer
        # keeps them live in between. Started on enter() and stopped on exit(), so nothing polls
        # (locally or on a remote server) while the module is not shown.
        self._resources_timer = QTimer(self)
        self._resources_timer.setInterval(3000)
        self._resources_timer.timeout.connect(self._update_resources)
        self.ui.remoteServerComboBox.setStyleSheet("""
            QComboBox[status="ok"]  { color: rgb(0,160,0); }
            QComboBox[status="bad"] { color: rgb(200,0,0); }
            """)

    def on_choose_studio_executable(self) -> None:
        """Pin a specific ``konfai-studio`` executable (e.g. a venv/pixi one) over the auto-detected one."""
        from qt import QFileDialog

        current = self._studio_executable() or os.path.expanduser("~")
        path = QFileDialog.getOpenFileName(self, "Select the konfai-studio executable", current)
        if not path or not os.path.isfile(path):
            return
        QSettings().setValue("KonfAI-Settings/StudioExecutable", path)
        self.update_logs(f"[KonfAI] Studio executable set to {path}", False)

    def on_stop_studio(self) -> None:
        """Ask the local Studio server to shut down (it runs detached, so there is no terminal)."""
        import urllib.request

        if not self._studio_answers():
            self.update_logs("[KonfAI] Studio is not running.", False)
            return
        try:
            urllib.request.urlopen(
                urllib.request.Request("http://127.0.0.1:8730/api/quit", method="POST"), timeout=3
            )
            self.update_logs("[KonfAI] Studio server stopped.", False)
        except Exception as exc:  # noqa: BLE001 - surfaced in the log, nothing else to do
            self.update_logs(f"[KonfAI] Could not stop Studio: {exc}", False)

    def _studio_answers(self) -> bool:
        import socket

        with socket.socket() as probe:
            probe.settimeout(0.25)
            return probe.connect_ex(("127.0.0.1", 8730)) == 0

    def _studio_env(self, executable: str) -> dict:
        """The environment to launch the Studio server with.

        A konfai-studio installed INSIDE Slicer's Python must keep Slicer's runtime variables: its
        shebang runs ``python-real``, which needs Slicer's LD_LIBRARY_PATH just to load libpython.
        One installed OUTSIDE (pipx, venv, pixi) gets the user's pre-Slicer environment instead
        (``startupEnvironment()``), so Slicer's rewritten libs do not leak into it. Either way,
        ``~/.local/bin`` joins the PATH: that is where the ``claude`` CLI — Studio's default
        brain — lives, and Slicer's PATH rewrite is exactly what hid it.
        """
        inside_slicer = executable.startswith(slicer.app.slicerHome)
        env = dict(os.environ) if inside_slicer else dict(slicer.util.startupEnvironment())
        user_bin = os.path.expanduser("~/.local/bin")
        path = env.get("PATH", "")
        if user_bin not in path.split(os.pathsep):
            env["PATH"] = f"{path}{os.pathsep}{user_bin}" if path else user_bin
        return env

    def _studio_executable(self) -> str | None:
        """The ``konfai-studio`` CLI: the user-remembered path first, then PATH, then ~/.local/bin,
        then Slicer's own scripts directory (not on PATH — where the one-click install puts it)."""
        import shutil
        import sysconfig

        stored = QSettings().value("KonfAI-Settings/StudioExecutable")
        if stored and os.path.isfile(stored):
            return str(stored)
        scripts = sysconfig.get_path("scripts")
        candidates = [
            shutil.which("konfai-studio"),
            os.path.expanduser("~/.local/bin/konfai-studio"),
            os.path.join(scripts, "konfai-studio"),
            os.path.join(scripts, "konfai-studio.exe"),
        ]
        return next((c for c in candidates if c and os.path.isfile(c)), None)

    def on_open_studio(self) -> None:
        """Open KonfAI Studio in the browser; asks before starting its local server when none runs.

        The server (loopback, port 8730) is launched detached, so it survives Slicer. When the CLI
        is not on Slicer's PATH (e.g. it lives in a virtualenv/pixi env), the user can locate the
        executable once; the choice is remembered in the settings.
        """
        import subprocess
        import time

        from qt import QDesktopServices, QFileDialog, QMessageBox, QUrl

        if self._studio_answers():
            self.update_logs("[KonfAI] Studio is already running — opening http://127.0.0.1:8730", False)
            QDesktopServices.openUrl(QUrl("http://127.0.0.1:8730"))
            return

        if (
            QMessageBox.question(
                self,
                "Launch KonfAI Studio?",
                "KonfAI Studio is the local web app for designing and running experiments.\n\n"
                "This will start its server on this machine (127.0.0.1:8730, not exposed to the\n"
                "network) and open it in your default browser. The server keeps running in the\n"
                "background after Slicer closes.\n\nLaunch it now?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            != QMessageBox.Yes
        ):
            return

        executable = self._studio_executable()
        if executable is None:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Install KonfAI Studio?")
            box.setText("KonfAI Studio is not installed yet.")
            box.setInformativeText(
                "Install it now (one click, about a minute)?\n\n"
                "Developers can instead locate an existing 'konfai-studio' executable "
                "(e.g. inside a virtualenv or pixi environment)."
            )
            install = box.addButton("Install", QMessageBox.AcceptRole)
            locate = box.addButton("Locate…", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() == install:
                with slicer_wait_popup("KonfAI Studio", "Installing konfai-studio..."):
                    slicer.util.pip_install("konfai-studio")
                executable = self._studio_executable()
                if executable is None:
                    QMessageBox.warning(
                        self,
                        "KonfAI Studio",
                        "The install finished but the 'konfai-studio' command was not found.",
                    )
                    return
            elif box.clickedButton() == locate:
                path = QFileDialog.getOpenFileName(
                    self, "Select the konfai-studio executable", os.path.expanduser("~")
                )
                if not path or not os.path.isfile(path):
                    return
                QSettings().setValue("KonfAI-Settings/StudioExecutable", path)
                executable = path
            else:
                return

        self.update_logs(f"[KonfAI] Starting Studio: {executable}", False)
        subprocess.Popen(  # noqa: S603 - launching the user's own CLI, detached on purpose
            [executable],
            env=self._studio_env(executable),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(32):  # give uvicorn a moment to bind before the browser tab loads
            if self._studio_answers():
                break
            slicer.app.processEvents()
            time.sleep(0.25)
        else:
            QMessageBox.warning(
                self,
                "KonfAI Studio",
                "The Studio server did not answer on 127.0.0.1:8730 after 8 seconds.\n"
                "Run 'konfai-studio' in a terminal to see why it fails.",
            )
            return
        self.update_logs("[KonfAI] Studio started — opening http://127.0.0.1:8730", False)
        QDesktopServices.openUrl(QUrl("http://127.0.0.1:8730"))

    def _set_remote_server_item_color(self, ok: bool) -> None:
        combo = self.ui.remoteServerComboBox
        combo.setProperty("status", "ok" if ok else "bad")
        combo.style().unpolish(combo)
        combo.style().polish(combo)
        combo.update()

    def on_remote_server_changed(self):
        remote_server = self.ui.remoteServerComboBox.currentData

        was_blocked = self.ui.deviceComboBox.blockSignals(True)
        self.ui.deviceComboBox.clear()
        self.ui.deviceComboBox.blockSignals(was_blocked)
        self.ui.remoteServerConfigButton.setEnabled(remote_server is not None)

        if remote_server is not None:
            from konfai import check_server

            ok, msg = check_server(remote_server)
            self._set_remote_server_item_color(ok)
            if not ok:
                if self._parameter_node is not None:
                    self._parameter_node.SetParameter("Device", "None")
                    self._parameter_node.SetParameter(
                        "RemoteServer",
                        f"{remote_server}|False",
                    )
                return
        else:
            self._set_remote_server_item_color(True)

        available_devices = self._get_available_devices()
        was_blocked = self.ui.deviceComboBox.blockSignals(True)
        for available_device in available_devices:
            self.ui.deviceComboBox.addItem(available_device[0], available_device[1])
        self.ui.deviceComboBox.blockSignals(was_blocked)

        index = -1
        if self._parameter_node:
            devices = self._parameter_node.GetParameter("Device")
            if devices == "None":
                index = 0
            else:
                index = self.ui.deviceComboBox.findData(devices.split(","))

        if index == -1:
            index = self.ui.deviceComboBox.count - 1
        self.ui.deviceComboBox.setCurrentIndex(index)
        if index == 0:
            self.on_device_changed()

        if self._parameter_node is not None:
            self._parameter_node.SetParameter(
                "RemoteServer",
                f"{self.ui.remoteServerComboBox.currentData}|True",
            )

        self._update_ram()
        if self._current_konfai_app:
            self._current_konfai_app.on_remote_server_changed()

    def on_remote_server_config(self):
        index = self.ui.remoteServerComboBox.currentIndex
        if index < 0:
            return

        remote_server = self.ui.remoteServerComboBox.currentData

        dlg = RemoteServerConfigDialog(remote_server)
        if dlg.exec_() != QDialog.Accepted:
            return

        if dlg.want_remove():
            self.ui.remoteServerComboBox.removeItem(index)
            return

        self.ui.remoteServerComboBox.setItemData(index, dlg.get())
        self.on_remote_server_changed()

    def on_add_remote_server(self):
        remote_servers_name = [
            self.ui.remoteServerComboBox.itemText(i) for i in range(self.ui.remoteServerComboBox.count)
        ]

        dlg = RemoteServerAddDialog(remote_servers_name)
        if dlg.exec_() != QDialog.Accepted:
            return

        remote_server = dlg.get()

        # update combobox
        self.ui.remoteServerComboBox.addItem(remote_server.name, remote_server)
        self.ui.remoteServerComboBox.setCurrentText(remote_server.name)

    def register_apps(self, apps: list[AppTemplateWidget]) -> None:
        """
        Register one or multiple KonfAI application widgets.

        If multiple apps are provided, a QTabWidget is created with one tab per app.
        If a single app is provided, it is inserted directly into the layout.

        Parameters
        ----------
        apps : list[AppTemplateWidget]
            App widgets to register.
        """
        if len(apps) > 1:
            tab_widget = QTabWidget()
            for app in apps:
                tab_widget.addTab(app, app._name)

            def on_tab_changed(index: int) -> None:
                """
                Called when the user switches between KonfAI app tabs.
                """
                current_app = tab_widget.widget(index)
                if current_app:
                    self._current_konfai_app = current_app
                    current_app.enter()

            tab_widget.currentChanged.connect(on_tab_changed)
            self.KonfAICoreWidget.layout().insertWidget(1, tab_widget)
        else:
            # Single app case: directly insert widget into layout
            app = apps[0]
            self.KonfAICoreWidget.layout().insertWidget(1, app)
            self._current_konfai_app = app

        # Initialize each app with shared callbacks and parameter node
        for app in apps:
            self._apps[app._name] = app
            app.app_setup(
                self.update_logs,
                self.update_progress,
                self._parameter_node,
                self.begin_transient_progress_feedback,
                self.end_transient_progress_feedback,
            )

        # Enter the first app by default
        app = next(iter(self._apps.values()))
        self._current_konfai_app = app

    def initialize_parameter_node(self):
        """
        Ensure a parameter node exists and is observed.

        This method also initializes a few global parameters such as 'is_running'.
        """
        # Unobserve previously selected parameter node
        if self._parameter_node is not None:
            self.removeObserver(self._parameter_node, vtk.vtkCommand.ModifiedEvent, self.update_gui_from_parameter_node)

        self._parameter_node = self.getParameterNode()

        # Observe the newly selected parameter node
        if self._parameter_node is not None:
            self.addObserver(self._parameter_node, vtk.vtkCommand.ModifiedEvent, self.update_gui_from_parameter_node)

        if self._parameter_node is not None:
            self._parameter_node.SetParameter("is_running", "False")
        self.update_gui_from_parameter_node()

    def update_gui_from_parameter_node(self, caller=None, event=None) -> None:
        """
        Update GUI elements when the parameter node changes.

        This delegates the GUI update to the currently active KonfAI app
        and also updates the availability of the work directory button.
        """
        if self._parameter_node and not self._updatingGUIFromParameterNode and self._current_konfai_app:
            self._updatingGUIFromParameterNode = True
            self._current_konfai_app.update_gui_from_parameter_node()
            self._updatingGUIFromParameterNode = False

            # Enable work directory button only when an app has an active work dir
            self.ui.openTempButton.setEnabled(self._current_konfai_app.get_work_dir() is not None)
            # self.on_remote_server_changed()

    def on_scene_start_close(self, caller, event) -> None:
        """
        Called just before the MRML scene is closed.

        The parameter node will be reset, so we temporarily drop reference to it.
        """
        if self._parameter_node is not None:
            self.removeObserver(self._parameter_node, vtk.vtkCommand.ModifiedEvent, self.update_gui_from_parameter_node)

        self._parameter_node = None

    def on_scene_end_close(self, caller, event) -> None:
        """
        Called just after the MRML scene is closed.

        If this module is visible while the scene is closed, we recreate a new parameter node.
        """
        self.initialize_parameter_node()

    def _get_available_devices(self) -> list[tuple[str, list[int] | None]]:
        """
        Return the list of available computation devices.

        The list always includes a CPU entry and, when possible, combinations
        of GPUs detected via PyTorch and NVML.

        Returns
        -------
        list[tuple[str, str | None]]
            List of (display_label, device_index_string or None) tuples.
        """
        # CPU fallback
        available_devices: list[tuple[str, list[int] | None]] = [("cpu [slow]", None)]
        remote_server = self.ui.remoteServerComboBox.currentData

        from konfai import get_available_devices

        device_index, device_name = get_available_devices(remote_server)
        devices = [(device_index, device_name) for device_index, device_name in zip(device_index, device_name)]

        combos: list[Any] = []
        # Build combinations of GPU indices, so multi-GPU usage can be exposed
        for r in range(1, len(devices) + 1):
            combos.extend(itertools.combinations(devices, r))
        for device in combos:
            devices_index_str = ",".join([str(index) for index, _ in device])
            devices_name = ",".join([name for _, name in device])
            available_devices.append((f"gpu {devices_index_str} - {devices_name}", [index for index, _ in device]))
        return available_devices

    def on_device_changed(self) -> None:
        """
        Called when the user changes the selected device (CPU/GPU).

        It updates VRAM monitoring and writes the selected device index
        into the parameter node.
        """
        self._update_vram()
        if self._parameter_node is not None:
            self._parameter_node.SetParameter(
                "Device",
                (
                    ",".join([str(index) for index in self.ui.deviceComboBox.currentData])
                    if self.ui.deviceComboBox.currentData
                    else "None"
                ),
            )

    def on_open_work_dir(self) -> None:
        """
        Open the current KonfAI app working directory in the file browser.
        """
        if self._current_konfai_app and self._current_konfai_app.get_work_dir():
            work_dir = self._current_konfai_app.get_work_dir()
            opened, error = open_path_in_file_browser(work_dir)
            if not opened:
                slicer.util.errorDisplay(f"Could not open folder:\n{work_dir}", detailedText=error or "")

    def _update_resources(self) -> None:
        self._update_ram()
        self._update_vram()

    def _update_ram(self) -> None:
        """
        Update the RAM usage display and progress bar.

        If usage exceeds 80%, the progress bar color is set to red.
        """
        remote_server = self.ui.remoteServerComboBox.currentData
        from konfai import get_ram

        used_gb, total_gb = get_ram(remote_server)

        self.ui.ramLabel.text = _("RAM used: {used:.1f} GB / {total:.1f} GB").format(used=used_gb, total=total_gb)
        self.ui.ramProgressBar.value = used_gb / total_gb * 100
        self.ui.ramProgressBar.setStyleSheet(_resource_bar_qss(used_gb / total_gb))

    def _update_vram(self) -> None:
        """
        Update the VRAM usage display and progress bar for the selected GPU(s).

        VRAM monitoring is only available when a GPU device is selected and NVML
        is successfully initialized.
        """
        device = self.ui.deviceComboBox.currentData
        if device is not None:
            remote_server = self.ui.remoteServerComboBox.currentData
            from konfai import get_vram

            used_gb, total_gb = get_vram(device, remote_server)

            self.ui.gpuLabel.show()
            self.ui.gpuProgressBar.show()
            self.ui.gpuLabel.text = _("VRAM used: {used:.1f} GB / {total:.1f} GB").format(used=used_gb, total=total_gb)
            self.ui.gpuProgressBar.value = used_gb / total_gb * 100
            self.ui.gpuProgressBar.setStyleSheet(_resource_bar_qss(used_gb / total_gb))
        else:
            # Hide VRAM widgets when CPU is selected
            self.ui.gpuLabel.hide()
            self.ui.gpuProgressBar.hide()

    def update_logs(self, text: str, clear: bool = False) -> None:
        """
        Append or replace text in the log window.

        RAM and VRAM usage are updated at the same time, so the user always
        has an up-to-date view of system resources during processing.
        """
        self._update_resources()
        if clear:
            self.ui.logText.plainText = text
        else:
            self.ui.logText.appendPlainText(text)

    def begin_transient_progress_feedback(self) -> None:
        if hasattr(self, "_transient_progress_state") and self._transient_progress_state is not None:
            return
        self._transient_progress_state = (self.ui.progressBar.value, self.ui.speedLabel.text)

    def end_transient_progress_feedback(self) -> None:
        state = getattr(self, "_transient_progress_state", None)
        if state is None:
            return
        value, text = state
        self.ui.progressBar.value = value
        self.ui.speedLabel.text = text
        self._transient_progress_state = None

    def update_progress(self, value: int, speed: str | None = None) -> None:
        """
        Update the progress bar and speed label.

        Parameters
        ----------
        value : int
            Progress percentage (0–100).
        speed : float | str
            Human-readable speed information, e.g. "5.2 it/s".
        """
        self.ui.progressBar.value = value
        if speed:
            self.ui.speedLabel.text = _("{speed}").format(speed=speed)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.

        Delegates cleanup to each registered KonfAI app (e.g., to remove
        temporary working directories).
        """
        self._resources_timer.stop()
        self.removeObservers()

        settings = QSettings()
        data = {}
        for i in range(self.ui.remoteServerComboBox.count):
            rs = self.ui.remoteServerComboBox.itemData(i)

            if rs is not None:
                data[rs.name] = {
                    "host": rs.host,
                    "port": rs.port,
                }

        settings.setValue(
            "KonfAI-Settings/RemoteServers",
            json.dumps(data),
        )

        for app in self._apps.values():
            app.cleanup()

    def enter(self) -> None:
        if not install_torch():
            return
        if not install_konfai():
            return
        is_first = False
        if self.ui.remoteServerComboBox.count == 0:
            is_first = True
            was_blocked = self.ui.remoteServerComboBox.blockSignals(True)
            settings = QSettings()
            raw = settings.value("KonfAI-Settings/RemoteServers")
            self.ui.remoteServerComboBox.addItem("Localhost", None)

            if raw is not None:
                for name, d in json.loads(raw).items():
                    remote_server = RemoteServer(
                        name=name,
                        host=d["host"],
                        port=int(d["port"]),
                    )
                    self.ui.remoteServerComboBox.addItem(name, remote_server)
            self.ui.remoteServerComboBox.blockSignals(was_blocked)

        index = 0
        if self._parameter_node:
            remote_server = self._parameter_node.GetParameter("RemoteServer")
            if remote_server.split("|")[0] != "None":
                index = self.ui.remoteServerComboBox.findText(remote_server.split("|")[0])

        if index == -1:
            index = 0
        if self.ui.remoteServerComboBox.currentIndex != index or is_first:
            self.ui.remoteServerComboBox.setCurrentIndex(index)
            if index == 0:
                self.on_remote_server_changed()

        if self._current_konfai_app:
            self._current_konfai_app.enter()

        self._update_resources()
        self._resources_timer.start()

    def exit(self) -> None:  # noqa: A003
        self._resources_timer.stop()
        if self._current_konfai_app:
            self._current_konfai_app.exit()
