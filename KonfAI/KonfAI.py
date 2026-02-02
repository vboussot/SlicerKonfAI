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

# flake8: noqa: E402
import itertools
import json
import os
import random
import re
import shutil
import time
import urllib.request
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
import vtk
from qt import (
    QCheckBox,
    QColor,
    QCursor,
    QDesktopServices,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFont,
    QFormLayout,
    QHBoxLayout,
    QIcon,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProcess,
    QPushButton,
    QSettings,
    QSize,
    QSpinBox,
    Qt,
    QTabWidget,
    QUrl,
    QVBoxLayout,
    QWidget,
)
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

SERVICE = "KonfAI"


def get_latest_pypi_version(package: str, timeout_s: float = 5.0) -> str | None:
    """
    Return the latest *released* version on PyPI for `package`, or None if unreachable.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["info"]["version"]
    except Exception as exc:
        print(f"[KonfAI] Could not query PyPI for {package}: {exc}")
        return None


def get_installed_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def ask_user_to_install_dependency(package_label: str, details: str) -> bool:
    """
    Ask the user for permission to install a dependency.
    Returns True if user accepts, False otherwise.
    """
    mb = QMessageBox(slicer.util.mainWindow())
    mb.setIcon(QMessageBox.Question)
    mb.setWindowTitle("Additional dependency required")
    mb.setText(f"This module requires {package_label}.")
    mb.setInformativeText(details + "\n\nIf you choose 'No', the module will close.")
    mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    mb.setDefaultButton(QMessageBox.Yes)
    return mb.exec_() == QMessageBox.Yes


@contextmanager
def slicer_wait_popup(title: str, text: str):
    """
    Show a modal 'please wait' dialog during long blocking operations.

    This prevents users from force-quitting Slicer because the UI looks frozen.
    """
    box = QMessageBox(slicer.util.mainWindow())
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QMessageBox.Information)
    box.setStandardButtons(QMessageBox.NoButton)  # no buttons = pure wait popup
    box.setModal(True)
    box.show()

    # Let Qt process paint events so the dialog is actually displayed
    slicer.app.processEvents()

    try:
        yield
    finally:
        box.hide()
        box.deleteLater()
        slicer.app.processEvents()


def install_torch() -> bool:
    try:
        import torch

        return True
    except ImportError:
        msg = (
            "SlicerKonfAI requires PyTorch to be installed in 3D Slicer using the official SlicerPyTorch extension.\n\n"
            "Step 1 – Install the SlicerPyTorch extension:\n"
            "  • In Slicer, open:  View → Extensions Manager\n"
            "  • Search for:  PyTorch (or SlicerPyTorch)\n"
            "  • Install the extension and restart Slicer when prompted.\n\n"
            "Step 2 – Install PyTorch using the SlicerPyTorch module:\n"
            "  • Open:  View → Modules\n"
            "  • Select:  Utilities → PyTorch\n"
            "  • Click 'Install PyTorch and choose the CPU or GPU build\n"
            "    that matches your system configuration.\n\n"
        )

        slicer.util.infoDisplay(msg, windowTitle="Prerequisite: PyTorch installation required")
        slicer.util.selectModule("Data")
        return False


def install_konfai() -> bool:
    package = "konfai"

    installed = get_installed_version(package)
    latest = get_latest_pypi_version(package)

    if latest is None:
        if installed is not None:
            print(f"[Konfai] installed ({installed}), PyPI unreachable -> keeping current.")
            return True
        if not ask_user_to_install_dependency(
            "KonfAI",
            "KonfAI is not installed.\n"
            "PyPI cannot be reached right now, but we can still try installing.\n"
            "Do you want to try?",
        ):
            slicer.util.selectModule("Data")
            return False
        with slicer_wait_popup("KonfAI dependency install", "Installing KonfAI..."):
            slicer.util.pip_install("konfai")
        return True

    if installed is None:
        if not ask_user_to_install_dependency(
            "KonfAI",
            f"KonfAI is required.\nLatest available version on PyPI: {latest}\n\n" "Do you want to install it now?",
        ):
            slicer.util.selectModule("Data")
            return False
        with slicer_wait_popup("KonfAI dependency install", f"Installing KonfAI {latest}..."):
            slicer.util.pip_install(f"konfai=={latest}")

        return True

    if installed != latest:
        if not ask_user_to_install_dependency(
            "KonfAI",
            f"A newer KonfAI version is available.\n"
            f"Installed: {installed}\n"
            f"Latest:    {latest}\n\n"
            "Do you want to upgrade now?",
        ):
            return True

        with slicer_wait_popup(
            "KonfAI dependency upgrade",
            f"Upgrading KonfAI from {installed} to {latest}...\n" "This may take several minutes.\n\n",
        ):
            slicer.util.pip_install(f"konfai=={latest}")
    if installed is not None:
        import konfai

        try:
            konfai.assert_konfai_install()
        except Exception as e:
            if not ask_user_to_install_dependency(
                "KonfAI",
                "KonfAI is installed but not functional (missing/broken dependencies).\n\n"
                "Do you want to reinstall KonfAI (and its dependencies) now?\n\n"
                f"Details:\n{e}",
            ):
                slicer.util.selectModule("Data")
                return False
            with slicer_wait_popup("KonfAI repair", "Reinstalling KonfAI and dependencies..."):

                slicer.util.pip_install(f"--upgrade --force-reinstall --no-deps konfai=={latest}")
                slicer.util.pip_install(
                    "tqdm numpy ruamel.yaml psutil tensorboard lxml h5py nvidia-ml-py requests huggingface_hub"
                )
            try:
                konfai.assert_konfai_install()
                return True
            except Exception as e2:
                slicer.util.errorDisplay("KonfAI was reinstalled but is still not functional.\n\n" f"{e2}")
                slicer.util.selectModule("Data")
                return False
    return True


def has_node_content(node) -> bool:
    if not node:
        return False

    if node.IsA("vtkMRMLVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode"):
        return bool(node.GetImageData())

    if node.IsA("vtkMRMLSegmentationNode"):
        seg = node.GetSegmentation()
        return seg is not None and seg.GetNumberOfSegments() > 0

    return False


class KonfAI(ScriptedLoadableModule):
    """
    Main Slicer module class for KonfAI.

    This class is responsible for registering the module in 3D Slicer,
    providing metadata (name, category, help text, acknowledgements, etc.),
    and making the module discoverable from the Slicer module list.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KonfAI")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Pipelines")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Valentin Boussot (University of Rennes, France)",
        ]
        # Short help text shown in the module panel
        self.parent.helpText = _(
            "<p>"
            "SlicerKonfAI is a 3D Slicer module that lets you run KonfAI Apps directly inside Slicer."
            "</p>"
            "<p>"
            "You can:<br>"
            "&bull; Discover and download KonfAI Apps hosted on Hugging Face<br>"
            "&bull; Run fast GPU-accelerated inference on volumes already loaded in Slicer<br>"
            "&bull; Automatically export/import data (DICOM, NRRD, NIfTI, etc.) as volumes or segmentations<br>"
            "&bull; Perform Quality Assurance (QA) using reference-based metrics or reference-free uncertainty "
            "estimation (TTA, MC dropout, multi-model ensembling)"
            "</p>"
            "<p>"
            "Each KonfAI App is a self-contained workflow "
            "(Prediction/Evaluation/Uncertainty YAML configs + trained model) "
            "that can be executed identically from Python, the CLI, or this Slicer module."
            "</p>"
            "<p><b>In short:</b><br>"
            "SlicerKonfAI = GUI + data exchange + process manager<br>"
            "KonfAI      = computation engine for training, inference, and evaluation."
            "</p>"
        )

        # Acknowledgment text (displayed in the About section)
        self.parent.acknowledgementText = _(
            "<p>This module was originally developed by Valentin Boussot "
            "(University of Rennes, France).<br>"
            "It integrates the KonfAI deep learning framework for medical imaging.</p>"
            "<p>If you use KonfAI in your research, please cite the following work:<br>"
            "Boussot V., Dillenseger J.-L.:<br>"
            "<b>KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.</b><br>"
            '<a href="https://arxiv.org/abs/2508.09823">https://arxiv.org/abs/2508.09823</a>'
            "</p>"
        )


def resource_path(filename: str) -> str:
    """
    Return the absolute path of a file located in the module's `Resources` directory.

    Parameters
    ----------
    filename : str
        File name relative to the `Resources` directory.

    Returns
    -------
    str
        Absolute path to the requested resource file.
    """
    scripted_modules_path = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scripted_modules_path, "Resources", filename)


class KonfAIMetricsPanel(QWidget):
    """
    Metrics panel widget for KonfAI QA.

    This panel displays metrics computed during evaluation or uncertainty analysis,
    and allows the user to quickly load and visualize corresponding images / volumes
    in the Slicer scene.
    """

    def __init__(self):
        super().__init__()

        # Load the associated .ui file and attach it to this widget
        ui_widget = slicer.util.loadUI(resource_path("UI/KonfAIMetricsPanel.ui"))
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Connect the image list widget to the click event handler
        self.ui.imagesListWidget.itemClicked.connect(self.on_image_clicked)

    def clear_metrics(self) -> None:
        """
        Clear the metrics list so no values are displayed in the panel.
        """
        self.ui.metricsListWidget.clear()

    def add_metric(self, key: str, value: float) -> None:
        """
        Add a single metric to the metrics list.

        Parameters
        ----------
        key : str
            Metric name, for example "Dice" or "MAE".
        value : float
            Metric value to be displayed.
        """
        # Format metric in an aligned and compact way using a monospace font
        text = f"{key:<15} : {value:.4g}"
        item = QListWidgetItem(text)
        font = QFont("Courier New", 10)
        item.setFont(font)
        self.ui.metricsListWidget.addItem(item)

    def set_metrics(self, metrics: dict[str, float]) -> None:
        """
        Replace the entire metric list with the metrics provided in `metrics`.

        Parameters
        ----------
        metrics : dict[str, float]
            Mapping between metric names and values.
        """
        for key, value in metrics.items():
            self.add_metric(key, value)

    def on_image_clicked(self) -> None:
        """
        Handle click on an image item in the list.

        Loads the corresponding volume from disk into Slicer,
        attaches metadata as node attributes, and sets it as the background volume
        in the slice viewers.
        """
        item = self.ui.imagesListWidget.currentItem()
        if not item:
            return

        full_path = item.data(Qt.UserRole)
        if not full_path:
            return

        # Load the volume node from disk
        volume_node = slicer.util.loadVolume(full_path)

        # Extract KonfAI-related metadata from the image file and attach them as node attributes
        from konfai.utils.dataset import get_infos

        _, attr = get_infos(full_path)
        for key, value in attr.items():
            # Keep only the part before '_' to have simple attribute keys
            volume_node.SetAttribute(key.split("_")[0], str(value))

        # Make the loaded volume visible in Slicer's slice views
        slicer.util.setSliceViewerLayers(background=volume_node)

    def clear_images_list(self) -> None:
        """
        Clear the list of image entries displayed in the panel.
        """
        self.ui.imagesListWidget.clear()

    def refresh_images_list(self, path: Path) -> None:
        """
        Populate the image list widget with all .mha files found under `path`.

        Parameters
        ----------
        path : Path
            Directory in which to recursively search for .mha images.
        """
        self.ui.imagesListWidget.clear()
        for filename in sorted(path.rglob("*.mha")):
            item = QListWidgetItem(filename.name)
            item.setFont(QFont("Arial", 10))
            # Store full path in the item for later retrieval in `on_image_clicked`
            item.setData(Qt.UserRole, str(filename))
            self.ui.imagesListWidget.addItem(item)


class Process(QProcess):
    """
    Thin wrapper around QProcess to handle KonfAI CLI processes.

    Responsibilities:
      - Forward stdout lines to the GUI log callback
      - Parse progress and speed from stdout and update the progress bar
      - Forward stderr lines to the console for debugging
    """

    def __init__(
        self,
        update_logs: Callable[[str, bool | None], None],
        update_progress: Callable[[int, str | None | None], None],
        running_setter: Callable[[bool], None],
    ):
        super().__init__(self)
        self.readyReadStandardOutput.connect(self.on_stdout_ready)
        self.readyReadStandardError.connect(self.on_stderr_ready)

        # Callbacks defined by the KonfAI core widget
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._running_setter = running_setter

    def on_stdout_ready(self) -> None:
        """
        Handle new data available on the standard output.

        This method:
          - Normalizes line endings
          - Forwards the message to the log window
          - Extracts numerical progress (0–100 %) and speed (e.g., '5 it/s')
            from the log line when present, and forwards them to the UI.
        """
        line = self.readAllStandardOutput().data().decode().strip()
        if line:
            # Keep only the last sub-line if carriage returns are used for progress overwrite
            line = line.replace("\r\n", "\n").split("\r")[-1]

            # Forward to the log callback
            self._update_logs(line, False)

            # Parse progress percentage if present (e.g., " 45% 123/456")
            m = re.search(r"\b(\d{1,3})%(?=\s+\d+/\d+)", line)
            # Parse speed pattern if present (e.g., "5.2 it/s" or "0.21 s/it")
            speed = re.search(r"([\d.]+)\s*(it/s|s/it)", line)

            if m:
                pct = int(m.group(1))
            else:
                pct = None

            if speed is not None and pct is not None:
                # Notify UI of updated progress and speed
                self._update_progress(pct, speed.group(1) + " " + speed.group(2))
            else:
                m = re.search(r"\b(\d{1,3})%\|", line)
                if m:
                    pct = int(m.group(1))
                    self._update_progress(pct, "")

    def on_stderr_ready(self) -> None:
        """
        Handle new data available on the standard error stream.

        Currently we simply print the content to the Python console for debugging.
        """

        print("Error : ", self.readAllStandardError().data().decode().strip())

    def run(self, command: str, work_dir: Path, args: list[str], on_end_function: Callable[[], None]) -> None:
        """
        Start a new subprocess with the given command and arguments.

        Parameters
        ----------
        command : str
            Executable name (e.g., 'konfai-apps').
        work_dir : Path
            Working directory in which the process will be executed.
        args : list[str]
            List of command-line arguments to pass to the executable.
        on_end_function : callable
            Callback to execute once the process has finished.
        """
        self.setWorkingDirectory(str(work_dir))

        t0 = time.perf_counter()

        # Disconnect any previous 'finished' slot to avoid stacking connections
        try:
            self.finished.disconnect()
        except TypeError:
            # No slot connected yet
            pass

        def _on_finished() -> None:
            if self.exitStatus() != QProcess.NormalExit:
                self._running_setter(False)
                return
            on_end_function()
            dt = time.perf_counter() - t0
            h, r = divmod(dt, 3600)
            m, s = divmod(r, 60)
            if h >= 1:
                msg = f"{int(h)}h {int(m)}m {s:.1f}s."
            elif m >= 1:
                msg = f"{int(m)}m {s:.1f}s."
            else:
                msg = f"{s:.2f}s."

            self._update_logs(f"Processing finished in {msg}", False)
            self._update_progress(100, None)
            self._running_setter(False)

        self.finished.connect(_on_finished)

        # Start the process asynchronously
        self.start(command, args)

    def stop(self) -> None:
        """
        Immediately terminate the running process, if any.
        """
        if self.state() == QProcess.ProcessState.NotRunning:
            return
        self.terminate()
        if not self.waitForFinished(2000):
            self.kill()
            self.waitForFinished(-1)


class RemoteServer:

    def __init__(self, name: str, host: str, port: int) -> None:
        self.name = name
        self.host = host
        self.port = port
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring

        self.token = keyring.get_password(SERVICE, str(self))
        self.timeout = 3

    def __str__(self) -> str:
        return f"{self.name}|{self.host}|{self.port}"

    def get_headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class DownloadFilesDialog(QDialog):
    def __init__(self, files: list[str], checkpoints_name_available: list[str]):
        super().__init__()
        self.setWindowTitle("Download from Hugging Face")
        self.setModal(True)
        self.resize(600, 450)

        self.label = QLabel("Select files to download:")
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.MultiSelection)

        for f in files:
            item = QListWidgetItem(f)

            if f.endswith(".pt") and f not in checkpoints_name_available:
                item.setForeground(QColor("#9ca3af"))
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setData(Qt.UserRole, "shadow")

            self.listw.addItem(item)

        # boutons (même style que ton exemple)
        self.downloadButton = QPushButton("Download")
        self.cancelButton = QPushButton("Cancel")

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.downloadButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addWidget(self.label)
        root.addWidget(self.listw)
        root.addLayout(btns)

        self.downloadButton.clicked.connect(lambda _=False: self.accept())
        self.cancelButton.clicked.connect(lambda _=False: self.reject())

    def selected_files(self) -> list[str]:
        return [i.text() for i in self.listw.selectedItems()]


class AppTemplateWidget(QWidget):
    """
    Abstract base widget for a KonfAI application panel.

    This class encapsulates common logic for:
      - Managing a temporary working directory
      - Managing process lifecycle and "Run/Stop" behavior
      - Synchronizing a Slicer parameter node with the GUI

    Child classes must implement methods to initialize/update parameter nodes
    and to propagate changes between GUI and parameter node.
    """

    def __init__(self, name: str, ui_widget):
        super().__init__()
        self._process: Process | None = None
        self._parameter_node = None
        self._work_dir: Path = None  # type: ignore[assignment]
        self._name = name

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Bind the MRML scene so that qMRML widgets in the .ui file work properly
        ui_widget.setMRMLScene(slicer.mrmlScene)
        self._initialized = False

    def app_setup(self, update_logs, update_progress, parameter_node) -> None:
        """
        Initialize the application with callbacks and parameter node.

        Parameters
        ----------
        update_logs : callable
            Function to call when a new log line should be displayed.
        update_progress : callable
            Function to call to update progress value and speed.
        parameterNode : vtkMRMLScriptedModuleNode
            Parameter node used to persist application state.
        """
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameter_node = parameter_node
        self.process = Process(update_logs, update_progress, self.set_running)

    def get_work_dir(self) -> Path | None:
        """
        Return the current working directory used for temporary files,
        or None if no working directory is currently defined.
        """
        return self._work_dir

    def create_new_work_dir(self) -> None:
        """
        Create a fresh temporary directory for the current operation.
        """
        self._work_dir = Path(slicer.util.tempDirectory())

    def remove_work_dir(self) -> None:
        """
        Delete the current temporary working directory and reset the reference.

        The directory is removed recursively if it exists.
        """
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir)
            self._work_dir = None  # type: ignore[assignment]

    def is_running(self) -> bool:
        """
        Check whether an operation is currently running according to the parameter node.

        Returns
        -------
        bool
            True if the parameter node reports that a process is running, False otherwise.
        """
        return self._parameter_node is not None and self._parameter_node.GetParameter("is_running") == "True"

    def set_parameter(self, key: str, value: str) -> None:
        """
        Set a standard parameter (string value) on the associated Slicer parameter node.
        Parameters are namespaced under this object's name (self._name/<key>).
        """
        if self._parameter_node is not None:
            self._parameter_node.SetParameter(f"{self._name}/{key}", str(value))

    def set_parameter_node(self, key: str, value) -> None:
        """
        Set a reference to another MRML node on the parameter node.
        The reference is stored under a namespaced key (self._name/<key>).
        'value' must be a valid MRML node ID.
        """
        if self._parameter_node is not None:
            self._parameter_node.SetNodeReferenceID(f"{self._name}/{key}", value)

    def get_parameter(self, key: str) -> str | bool:
        """
        Retrieve a parameter value stored under self._name/<key>.
        Returns a string if found, otherwise False.
        Useful for checking whether a parameter exists (truthy/falsey behavior).
        """
        if self._parameter_node is not None:
            return self._parameter_node.GetParameter(f"{self._name}/{key}")
        else:
            return False

    def get_parameter_node(self, key: str):
        """
        Retrieve a referenced MRML node stored under self._name/<key>.
        Returns the node object if found, otherwise None.
        """
        if self._parameter_node is not None:
            return self._parameter_node.GetNodeReference(f"{self._name}/{key}")
        else:
            return None

    def get_device(self) -> list[str]:
        """
        Convenience accessor to retrieve the associated 'Device' parameter.
        Returns the device string if set and not 'None', otherwise None.
        """
        if self._parameter_node is not None and self._parameter_node.GetParameter("Device") != "None":
            return self._parameter_node.GetParameter("Device").split(",")
        else:
            return []

    def get_remote_server(self) -> tuple[RemoteServer | None, bool]:
        if self._parameter_node is not None and self._parameter_node.GetParameter("RemoteServer") != "None|True":
            name, host, port, ok = self._parameter_node.GetParameter("RemoteServer").split("|")
            return RemoteServer(name, host, port) if host != "None" else None, eval(ok)
        else:
            return None, False

    def set_running(self, state: bool) -> None:
        """
        Update the 'is_running' flag in the parameter node.

        Parameters
        ----------
        state : bool
            True if an operation is considered running, False otherwise.
        """
        if self._parameter_node is not None:
            self._parameter_node.SetParameter("is_running", str(state))

    def on_run_button(self, function) -> None:
        """
        Generic handler for the "Run"/"Stop" button.

        If no process is running:
          - Clean and recreate a working directory
          - Mark the state as running
          - Initialize log and progress
          - Call the provided function

        If a process is already running:
          - Request termination of the process.
        """

        if not self.is_running():
            # Start a new operation
            remote_server, ok = self.get_remote_server()
            device = self.get_device()
            if remote_server is not None:
                from konfai import check_server

                new_ok, msg = check_server(remote_server)
                if ok != new_ok:
                    if self._parameter_node is not None:
                        self._parameter_node.SetParameter(
                            "RemoteServer",
                            f"{remote_server}|{new_ok}",
                        )
                    if new_ok:
                        QMessageBox.information(
                            self,
                            "Remote server available",
                            "The connection to the remote server has been restored.\n\n"
                            "Please select the appropriate GPU and restart the operation.",
                        )
                        return

                if not new_ok:
                    QMessageBox.warning(
                        self,
                        "Remote server unreachable",
                        f"{msg}.\n\n"
                        "This remote server cannot be reached.\n"
                        "Please edit the host/port or select another remote server.",
                    )
                    return
            self.remove_work_dir()
            self.create_new_work_dir()
            self.set_running(True)
            self._update_logs("Processing started.", True)
            self._update_progress(0, "0 it/s")
            try:
                function(remote_server, device)
            except Exception as e:
                # Log the exception for debugging and reset running state
                print(e)
                self.set_running(False)
        else:
            # Stop current operation
            self.set_running(False)
            self.process.stop()

    def cleanup(self) -> None:
        """
        Called when the user closes the module and the widget is destroyed.

        This method is responsible for cleanup tasks such as removing
        the temporary working directory.
        """
        self.remove_work_dir()

    @abstractmethod
    def initialize_parameter_node(self) -> None:
        """
        Ensure parameter node values are initialized with sensible defaults.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_gui_from_parameter_node(self) -> None:
        """
        Initialize GUI state from the parameter node.

        Called when the widget is first entered or when a new parameter node is assigned.
        """
        raise NotImplementedError

    @abstractmethod
    def update_gui_from_parameter_node(self) -> None:
        """
        Update GUI components to reflect the current parameter node values.

        This method should not modify the parameter node, only the GUI.
        """
        raise NotImplementedError

    @abstractmethod
    def update_parameter_node_from_gui(self, caller=None, event=None) -> None:
        """
        Update the parameter node to reflect the current GUI state.

        This method is usually connected to GUI signals.
        """
        raise NotImplementedError

    def enter(self) -> None:
        """
        Called when the user enters the corresponding application tab.

        Ensures the parameter node is initialized and the GUI reflects its current state.
        """
        self.initialize_parameter_node()
        self.initialize_gui_from_parameter_node()
        self.update_gui_from_parameter_node()

    def exit(self) -> None:
        pass

    @abstractmethod
    def on_remote_server_changed(self):
        raise NotImplementedError


class ChipSelector:
    """
    Generic "ComboBox -> removable chips" controller.

    Responsibilities:
      - Add a chip when user selects an item from combo
      - Remove chip on click, and put the item back in combo
      - Keep an optional spinbox in sync with number of selected chips
      - Notify optional callback on change

    Assumptions:
      - `container_layout` is a QLayout that may end with a spacer item
      - combo exposes itemText(i), addItem(text), removeItem(i), findText(text),
        and emits activated(int) (Qt signal)
    """

    def __init__(
        self,
        combo,
        container_layout,
        spinbox=None,
        min_selected: int = 0,
        combo_remove: bool = True,
        on_change: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._combo = combo
        self._layout = container_layout
        self._spinbox = spinbox
        self._min_selected = max(0, int(min_selected))
        self._on_change = on_change
        self._combo_remove = combo_remove
        self._combo.connect("activated(int)", self._on_combo_activated)
        if self._spinbox is not None:
            self._spinbox.setMinimum(self._min_selected)
            self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        self._all_items: list[str] = []
        self._available_items: list[str] = []

    def _on_spinbox_changed(self):
        sel = self.selected()
        if self._spinbox.value == len(sel):
            return
        if self._spinbox.value > len(sel):
            self._add(sorted(list(set(self._all_items) - set(sel)))[0])
        else:
            text = sel[-1]
            self._remove(text)

    def update(self, availables_item: list[str], all_items: list[str], pre_selected: list[str]):
        self._all_items = all_items
        self._available_items = availables_item
        self._combo.clear()
        if self._spinbox is not None:
            self._spinbox.setMaximum(len(all_items))
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, QPushButton):
                self._layout.removeWidget(widget)
                widget.deleteLater()

        for name in all_items:
            if name in pre_selected:
                self._add(name)
            else:
                self._combo.addItem(name)
        if len(self.selected()) == 0 and len(all_items) > 0:
            self._add(all_items[0])

    def selected(self) -> list[str]:
        """Return currently selected chip texts (in layout order)."""
        out: list[str] = []
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                out.append(w.text)
        return out

    def _add(self, text: str) -> None:
        """Add a chip (if not already selected) and remove it from combo."""
        if not text or text in self.selected():
            return

        btn = QPushButton(text)
        btn.flat = True
        btn.toolTip = f"Click to remove {text}"
        btn.minimumHeight = 20
        btn.maximumHeight = 24

        shadow = text not in self._available_items
        btn.setProperty("shadow", "1" if shadow else "0")
        btn.setStyleSheet(
            """
            QPushButton {
                color: #0b3d91;
                background-color: #edf3ff;
                border: 1px solid #0b3d91;
                border-radius: 12px;
                padding: 3px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #dce8ff; }

            QPushButton[shadow="1"] {
                color: #6b7280;
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                font-weight: 500;
            }
            QPushButton[shadow="1"]:hover { background-color: #e5e7eb; }
            QPushButton[shadow="1"]:pressed { background-color: #e5e7eb; }
            """
        )

        def _remove():
            self._remove(text)

        btn.clicked.connect(_remove)
        self._layout.insertWidget(self._insert_index_before_spacer(), btn)

        if self._combo_remove:
            idx = self._combo.findText(text)
            if idx != -1:
                self._combo.removeItem(idx)
        self._sync()

    def _remove(self, text: str) -> None:
        """Remove a chip (respecting min_selected) and add it back to combo."""
        if not text:
            return

        cur = self.selected()
        if len(cur) <= self._min_selected:
            return

        # remove chip widget
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if isinstance(w, QPushButton) and w.text == text:
                self._layout.removeWidget(w)
                w.deleteLater()
                break

        # add back to combo (avoid duplicates)
        if self._combo.findText(text) == -1:
            self._combo.addItem(text)

        self._sync()

    def _sync(self) -> None:
        sel = self.selected()
        if self._spinbox is not None:
            try:
                self._spinbox.setValue(len(sel))
            except Exception:
                pass
        if self._on_change is not None:
            self._on_change(sel)

    def _on_combo_activated(self, index: int) -> None:
        text = self._combo.itemText(index)
        self._add(text)

    def _insert_index_before_spacer(self) -> int:
        insert_index = self._layout.count()
        for i in range(self._layout.count()):
            if self._layout.itemAt(i).spacerItem() is not None:
                insert_index = i
                break
        return insert_index


class KonfAIAppTemplateWidget(AppTemplateWidget):
    """
    Concrete implementation of AppTemplateWidget for KonfAI applications.

    This widget provides:
      - Input/output volume selection for inference
      - QA capabilities (evaluation with reference, or uncertainty estimation)
      - App selection from Hugging Face repositories
      - Configuration access for KonfAI Apps (YAML files)
    """

    def __init__(self, name: str, konfai_repo_list: list[str]):
        super().__init__(name, slicer.util.loadUI(resource_path("UI/KonfAIAppTemplate.ui")))
        self._konfai_repo_list = konfai_repo_list
        # Attach metrics panels in both QA contexts: reference-based and reference-free
        self.evaluation_panel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluation_panel)
        self.uncertainty_panel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertainty_panel)

        self._description_expanded = False

        # Connect volume selectors to parameter node synchronization
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.outputVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.ui.segmentationShow3DButton.setSegmentationNode
        )
        self.ui.segmentationShow3DButton.setVisible(False)

        # Evaluation / uncertainty input selectors
        self.ui.inputVolumeEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.on_input_volume_evaluation_changed
        )
        self.ui.inputVolumeSequenceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.on_input_volume_evaluation_changed
        )

        # Reference and transform nodes
        self.ui.referenceVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.on_input_volume_evaluation_changed
        )
        self.ui.referenceMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.inputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        self.ui.ttaSpinBox.valueChanged.connect(self.on_tta_changed)
        self.ui.mcDropoutSpinBox.valueChanged.connect(self.on_mc_dropout_changed)

        # App selection and management
        self.ui.addAppButton.clicked.connect(self.on_add_app)
        self.ui.removeAppButton.clicked.connect(self.on_remove_app)

        # Configuration button (opens folder containing KonfAI YAML configs)
        self.ui.configButton.clicked.connect(self.on_open_config)

        self.ui.refreshAppsListButton.setIcon(QIcon(resource_path("Icons/refresh.png")))
        self.ui.refreshAppsListButton.setIconSize(QSize(18, 18))
        self.ui.refreshAppsListButton.clicked.connect(self.on_refresh_app)

        # Run buttons for inference and QA
        self.ui.runInferenceButton.clicked.connect(self.on_run_inference_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        # Description toggle and QA tab changes
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)
        self.ui.appComboBox.currentIndexChanged.connect(self.on_app_selected)

        self.ui.uncertaintyCheckBox.toggled.connect(self.on_uncertainty_toggled)
        
        self.chip_selector = ChipSelector(
            self.ui.checkpointsComboBox,
            self.ui.selectedCheckpointsWidget.layout(),
            self.ui.ensembleSpinBox,
            1,
            on_change=self.on_checkpoint_selected_change,
        )
        self.app_local_repositoy: list[str] = []

    def on_uncertainty_toggled(self, checked: bool) -> None:
        self.set_parameter("uncertainty", str(checked))

    def on_checkpoint_selected_change(self, checkpoints_selected: list[str]):
        self.set_parameter("checkpoints_name", ",".join(checkpoints_selected))

    def on_refresh_app(self):
        from konfai.utils.utils import AppRepositoryError

        try:
            self.populate_apps(True)
        except AppRepositoryError as e:
            slicer.util.errorDisplay(
                "Unable to refresh the list of applications.\n\n"
                "This may happen if you are offline, if the repository is not accessible, "
                "The application list has not been updated.",
                detailedText=getattr(e, "details", None) or str(e),
            )

    def on_tta_changed(self):
        self.set_parameter("number_of_tta", str(self.ui.ttaSpinBox.value))

    def on_mc_dropout_changed(self):
        self.set_parameter("number_of_mc_dropout", str(self.ui.mcDropoutSpinBox.value))

    def set_information(
        self,
        app: str | None = None,
        number_of_ensemble: int | None = None,
        number_of_tta: int | None = None,
        number_of_mc_dropout: int | None = None,
    ) -> None:
        """
        Update the app information summary panel (app name + ensemble/TTA/MC counts).

        When any field is None or not available, a placeholder is displayed.
        """
        self.ui.appSummaryValue.setText(f"App: {app}" if app else "App: N/A")
        self.ui.ensembleSummaryValue.setText(f"#{number_of_ensemble}" if number_of_ensemble else "#N/A")
        self.ui.ttaSummaryValue.setText(f"#{number_of_tta}" if number_of_ensemble and number_of_tta else "#N/A")
        self.ui.mcSummaryValue.setText(
            f"#{number_of_mc_dropout}" if number_of_ensemble and number_of_mc_dropout else "#N/A"
        )

    def on_input_volume_evaluation_changed(self, node) -> None:
        """
        Handler called when the evaluation input or stack input selection changes.

        It attempts to read metadata from the selected node (or its storage),
        updates the app information summary, and synchronizes the parameter node.
        """
        if node:
            storage = node.GetStorageNode()
            if storage:
                path = storage.GetFileName()
                if path and Path(path).exists():
                    from konfai.utils.dataset import get_infos

                    _, attr = get_infos(path)
                    if (
                        "App" in attr
                        and "NumberOfEnsemble" in attr
                        and "NumberOfTTA" in attr
                        and "NumberOfMCDropout" in attr
                    ):
                        self.set_information(
                            attr["App"],
                            attr["NumberOfEnsemble"],
                            attr["NumberOfTTA"],
                            attr["NumberOfMCDropout"],
                        )
                    else:
                        self.set_information()
                else:
                    self.set_information()
            else:
                # Fallback: read attributes directly from the node if available
                self.set_information(
                    node.GetAttribute("App"),
                    node.GetAttribute("NumberOfEnsemble"),
                    node.GetAttribute("NumberOfTTA"),
                    node.GetAttribute("NumberOfMCDropout"),
                )
        self.update_parameter_node_from_gui()

    def on_remote_server_changed(self):
        self.populate_apps()

    def populate_apps(self, force_update: bool = False) -> None:
        remote_server, ok = self.get_remote_server()
        from konfai.utils.utils import (
            AppRepositoryError,
            get_app_repository_info,
            get_available_apps_on_hf_repo,
            get_available_apps_on_remote_server,
        )

        apps_name: list[str] = []
        if remote_server is None:
            settings = QSettings()
            raw = settings.value(f"KonfAI-Settings/{self._name}/Apps")
            apps_name = []
            if raw is not None:
                apps_name = json.loads(raw)

            default_apps_name = []

            for konfai_repo in self._konfai_repo_list:
                try:
                    default_apps_name += [
                        konfai_repo + ":" + app_name
                        for app_name in get_available_apps_on_hf_repo(konfai_repo, force_update)
                    ]
                except AppRepositoryError as e:
                    slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
                    return
            apps_name = list(set(default_apps_name + apps_name))
            self.app_local_repositoy = apps_name
        elif ok:
            apps_name = [
                f"{remote_server.host}:{remote_server.port}:{app_name}|{remote_server.token}"
                for app_name in get_available_apps_on_remote_server(remote_server)
            ]

        # Populate the app combo box with apps found in the provided Hugging Face repos or available app on remote server
        apps = []
        try:
            for app_name in sorted(apps_name):
                apps.append(get_app_repository_info(app_name, False))
        except Exception as e:
            slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
            return

        was_blocked = self.ui.appComboBox.blockSignals(True)
        self.ui.appComboBox.clear()
        for app in apps:
            self.ui.appComboBox.addItem(app.get_display_name(), app)
        self.ui.appComboBox.blockSignals(was_blocked)
        app_param = self.get_parameter("App")
        index = 0
        for i in range(self.ui.appComboBox.count):
            app = self.ui.appComboBox.itemData(i)
            if app.get_name() == app_param:
                index = i
                break
        self.ui.appComboBox.setCurrentIndex(index)

    def enter(self) -> None:
        """
        Overridden AppTemplateWidget entry point.

        Re-initializes parameter node, GUI state and ensures app selection
        is consistent when the widget is shown.
        """
        if self.ui.appComboBox.count == 0:
            self.populate_apps()

        super().enter()
        self.on_app_selected()

    def initialize_parameter_node(self) -> None:
        """
        Initialize the parameter node with default values for this app
        (input volume, app, ensemble/TTA/MC-dropout parameters).
        """
        self._initialized = False

        # Select default input nodes if nothing is selected yet
        if self.get_parameter_node("InputVolume") is None:
            first_volume_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if first_volume_node and self._parameter_node is not None:
                self._parameter_node.SetNodeReferenceID(f"{self._name}/InputVolume", first_volume_node.GetID())

        # Set default app if none is stored yet
        if not self.get_parameter("App"):
            app = self.ui.appComboBox.itemData(0)
            if app:
                self.set_parameter("App", app.get_name())

        # Determine the current app object from the stored parameter if possible
        current_app = None
        for i in range(self.ui.appComboBox.count):
            app_tmp = self.ui.appComboBox.itemData(i)
            if app_tmp.get_name() == self.get_parameter("App"):
                current_app = app_tmp
                break
        if current_app is None:
            current_app = self.ui.appComboBox.itemData(0)
        # Ensemble / TTA / MC-dropout defaults based on app capabilities
        if not self.get_parameter("number_of_tta"):
            self.set_parameter("number_of_tta", str(current_app.get_maximum_tta()) if current_app else "0")
        if not self.get_parameter("uncertainty"):
            self.set_parameter("uncertainty", "False")
        if not self.get_parameter("number_of_mc_dropout"):
            self.set_parameter("number_of_mc_dropout", str(current_app.get_mc_dropout()) if current_app else "0")
        self.initialize_gui_from_parameter_node()
        self._initialized = True

    def initialize_gui_from_parameter_node(self) -> None:
        """
        Initialize GUI widget values from the parameter node.
        """
        # App selection
        app_param = self.get_parameter("App")
        # Search the combo box items for a matching app name
        index = -1
        for i in range(self.ui.appComboBox.count):
            app = self.ui.appComboBox.itemData(i)
            if app.get_name() == app_param:
                index = i
                break
        self.ui.appComboBox.setCurrentIndex(index if index != -1 else 0)

        # Ensemble / TTA / MC-dropout spin boxes
        self.ui.ttaSpinBox.setValue(int(self.get_parameter("number_of_tta")))
        self.ui.mcDropoutSpinBox.setValue(int(self.get_parameter("number_of_mc_dropout")))
        self.ui.uncertaintyCheckBox.setChecked(self.get_parameter("uncertainty") == "True")

        # Input volume
        self.ui.inputVolumeSelector.setCurrentNode(self.get_parameter_node("InputVolume"))

        # Output volume
        self.ui.outputVolumeSelector.setCurrentNode(self.get_parameter_node("OutputVolume"))

        self.ui.referenceVolumeSelector.setCurrentNode(self.get_parameter_node("ReferenceVolume"))
        self.ui.inputVolumeEvaluationSelector.setCurrentNode(self.get_parameter_node("InputVolumeEvaluation"))

        self.ui.referenceMaskSelector.setCurrentNode(self.get_parameter_node("ReferenceMask"))

        self.ui.inputTransformSelector.setCurrentNode(self.get_parameter_node("InputTransform"))

        self.ui.inputVolumeSequenceSelector.setCurrentNode(self.get_parameter_node("InputVolumeSequence"))

    def update_gui_from_parameter_node(self) -> None:
        # Update run/stop label based on running state
        if not self.is_running():
            self.ui.runInferenceButton.text = "Run"
            self.ui.runEvaluationButton.text = "Run"
            self.ui.configButton.enabled = True
        else:
            self.ui.runInferenceButton.text = "Stop"
            self.ui.runEvaluationButton.text = "Stop"
            self.ui.configButton.enabled = False
            return

        """
        Update the GUI state based on the current parameter node values.

        This includes enabling/disabling buttons, updating tooltips and
        configuring default output volume names.
        """
        input_volume = self.get_parameter_node("InputVolume")
        if (
            has_node_content(input_volume)
            and self.ui.appComboBox.currentData
            and len(self.ui.appComboBox.currentData.get_checkpoints_name()) > 0
        ):
            self.ui.runInferenceButton.toolTip = _("Start inference")
            self.ui.runInferenceButton.enabled = True
        else:
            self.ui.runInferenceButton.toolTip = _("Select input volume")
            self.ui.runInferenceButton.enabled = False

        reference_volume = self.get_parameter_node("ReferenceVolume")
        input_evaluation_volume = self.get_parameter_node("InputVolumeEvaluation")
        inference_stack_volume = self.get_parameter_node("InputVolumeSequence")

        # Configure QA button depending on which tab is active
        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            # Reference-based evaluation: need both input and reference volumes
            if has_node_content(reference_volume) and has_node_content(input_evaluation_volume):
                self.ui.runEvaluationButton.toolTip = _("Start evaluation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input and reference volumes")
                self.ui.runEvaluationButton.enabled = False
        else:
            # Uncertainty estimation: need a stack volume with more than one component
            if inference_stack_volume and inference_stack_volume.GetNumberOfDataNodes() > 1:
                self.ui.runEvaluationButton.toolTip = _("Start uncertainty estimation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input volume")
                self.ui.runEvaluationButton.enabled = False

        # Suggest an output volume base name derived from input volume name
        if input_volume:
            self.ui.outputVolumeSelector.baseName = _("{volume_name} Output").format(volume_name=input_volume.GetName())

    def update_parameter_node_from_gui(self, caller=None, event=None) -> None:
        """
        Propagate current GUI selection values to the parameter node.

        This method should be called whenever a relevant widget changes
        (volume selectors, app selection, etc.).
        """
        if self._parameter_node is None or not self._initialized:
            return
        was_modified = self._parameter_node.StartModify()
        self.set_parameter_node("InputVolume", self.ui.inputVolumeSelector.currentNodeID)

        self.set_parameter_node("OutputVolume", self.ui.outputVolumeSelector.currentNodeID)

        self.set_parameter_node("ReferenceVolume", self.ui.referenceVolumeSelector.currentNodeID)
        self.set_parameter_node("InputVolumeEvaluation", self.ui.inputVolumeEvaluationSelector.currentNodeID)

        self.set_parameter_node("ReferenceMask", self.ui.referenceMaskSelector.currentNodeID)
        self.set_parameter_node("InputTransform", self.ui.inputTransformSelector.currentNodeID)

        self.set_parameter_node(
            "InputVolumeSequence",
            self.ui.inputVolumeSequenceSelector.currentNodeID,
        )
        self._parameter_node.EndModify(was_modified)

    def on_app_selected(self) -> None:
        """
        Handle app selection changes.

        This method:
          - Resets description expansion
          - Adjusts ensemble / TTA / MC-dropout spin box ranges
          - Enables/disables QA options depending on app capabilities
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
            self.ui.removeAppButton.setEnabled(app.get_name().split(":")[0] not in self._konfai_repo_list)
            self.ui.addAppButton.setEnabled(True)
            self.ui.configButton.setEnabled(True)

        from konfai.utils.utils import LocalAppRepositoryFromDirectory

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

        # Update ensemble / TTA / MC-dropout limits
        checkpoints_name_param = self.get_parameter("checkpoints_name")
        checkpoints_name = []
        if checkpoints_name_param and isinstance(checkpoints_name_param, str):
            checkpoints_name = checkpoints_name_param.split(",")
        self.chip_selector.update(app.get_checkpoints_name_available(), app.get_checkpoints_name(), checkpoints_name)
        if not self.get_parameter("number_of_tta") or int(self.get_parameter("number_of_tta")) > app.get_maximum_tta():
            self.set_parameter("number_of_tta", str(app.get_maximum_tta()))

        if (
            not self.get_parameter("number_of_mc_dropout")
            or int(self.get_parameter("number_of_mc_dropout")) > app.get_mc_dropout()
        ):
            self.set_parameter("number_of_mc_dropout", str(app.get_mc_dropout()))

        self.ui.ttaSpinBox.setEnabled(app.get_maximum_tta() > 0)
        self.ui.ttaSpinBox.setMaximum(app.get_maximum_tta())

        self.ui.mcDropoutSpinBox.setEnabled(app._mc_dropout > 0)
        self.ui.mcDropoutSpinBox.setMaximum(app._mc_dropout)

        # Enable QA sections based on app capabilities (evaluation/uncertainty support)
        has_inference, has_evaluation, has_uncertainty = app.has_capabilities()
        self.ui.evaluationCollapsible.setEnabled(has_evaluation or has_uncertainty)
        if not has_evaluation and not has_uncertainty:
            self.ui.evaluationCollapsible.collapsed = True

        # Enable/disable tabs
        self.ui.qaTabWidget.setTabEnabled(0, has_evaluation)
        self.ui.qaTabWidget.setTabEnabled(1, has_uncertainty)

        self.set_parameter("App", app.get_name())

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
        from konfai.utils.utils import LocalAppRepositoryFromDirectory

        app = self.ui.appComboBox.currentData

        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("Remove app?")
        mb.setText(f"Do you really want to remove “{app.get_display_name()}” from the list?")
        mb.setInformativeText("This will remove the app entry from the extension’s list.")
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)

        chk = QCheckBox("Also delete the folder from disk")
        chk.setChecked(False)
        mb.setCheckBox(chk)

        if mb.exec_() != QMessageBox.Yes:
            return

        index = self.ui.appComboBox.findText(app.get_display_name())
        self.ui.appComboBox.removeItem(index)
        self.app_local_repositoy.remove(app.get_name())

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
        from konfai.utils.utils import LocalAppRepositoryFromHF

        if isinstance(app, LocalAppRepositoryFromHF):
            filenames = LocalAppRepositoryFromHF.get_filenames(app._repo_id, app._app_name, True)
            dlg = DownloadFilesDialog(filenames, app.get_checkpoints_name_available())
            if dlg.exec() != dlg.Accepted:
                return
            selected_files = dlg.selected_files()
            if not selected_files:
                return

            import sys
            import textwrap

            pycode = textwrap.dedent(
                f"""
                from konfai.utils.utils import LocalAppRepositoryFromHF, MinimalLog
                with MinimalLog() as log:
                    filenames = {selected_files!r}
                    for filename in filenames:
                        LocalAppRepositoryFromHF.download(
                            "{app._repo_id}",
                            "{app._app_name}" + "/" + filename,
                            True,
                        )
                        print(f"[KonfAI-Apps] {{filename}} is ready.")
                """
            )

            def on_end_function() -> None:
                from konfai.utils.utils import get_app_repository_info

                idx = self.ui.appComboBox.currentIndex
                app = self.ui.appComboBox.currentData
                app = get_app_repository_info(app.get_name(), False)
                self.ui.appComboBox.setItemData(idx, app)
                self.on_app_selected()

            self._update_logs("Starting download...", True)
            self._update_progress(0, "")
            self.set_running(True)
            self.ui.runInferenceButton.enabled = True
            self.ui.runInferenceButton.toolTip = _("Stop download")
            self.process.run(sys.executable, Path("./").resolve(), ["-c", pycode], on_end_function)
        else:
            config_files = app.download_config_file()
            QDesktopServices.openUrl(QUrl.fromLocalFile(config_files[0].parent))

    def on_add_app(self) -> None:
        """
        Show a menu to add a new app (from a folder, from Hugging Face, or configure fine-tuning).
        """
        m = QMenu()
        act_folder = m.addAction("Add from folder…")
        act_hf = m.addAction("Add from Hugging Face…")
        act_ft = m.addAction("Setup fine-tuning")
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
        from konfai.utils.utils import LocalAppRepositoryFromDirectory

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

    def on_add_hf(self) -> None:
        """
        Add a app from a Hugging Face repository.

        The user is prompted for a repo id (e.g., 'VBoussot/ImpactSynth') and then
        a subdirectory (app name) is selected among the available directories.
        """
        from huggingface_hub import HfApi
        from konfai.utils.utils import LocalAppRepositoryFromHF

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
            files = api.list_repo_files(
                repo_id=base_repo_id,
                revision=revision or None,
                repo_type="model",
            )
        except Exception:
            QMessageBox.critical(None, "Hugging Face", f"Repository '{repo_id}' does not exist on Hugging Face.")
            return

        # List top-level directories as candidate app names
        dirs = sorted({path.split("/")[0] for path in files if "/" in path})
        app_name = self.ask_subdir(dirs)
        if app_name:
            from konfai.utils.utils import is_app_repo

            state, error, _ = is_app_repo(repo_id, app_name)
            if not state:
                QMessageBox.critical(None, "App", error)
                return

            app = LocalAppRepositoryFromHF(repo_id, app_name)

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
        from konfai.utils.utils import LocalAppRepositoryFromDirectory

        app = self.ui.appComboBox.currentData
        if app is None:
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

        app.install_fine_tune("Config.yml", Path(parent_dir) / name, display_name, epochs, it_validation)
        # Add the folder to the app combo box
        app_ft = LocalAppRepositoryFromDirectory(Path(parent_dir), name)
        self.ui.appComboBox.addItem(app_ft.get_display_name(), app_ft)
        self.ui.appComboBox.setCurrentIndex(self.ui.appComboBox.findData(app_ft))
        self.app_local_repositoy.append(app_ft.get_name())

    def on_tab_changed(self) -> None:
        """
        Update GUI state when the user switches between QA tabs.

        Ensures that button enabling/disabling is consistent with the current tab.
        """
        self.update_gui_from_parameter_node()

    def on_run_inference_button(self) -> None:
        """
        Run or stop inference depending on the current state.
        """
        self.on_run_button(self.inference)

    def on_run_evaluation_button(self) -> None:
        """
        Run or stop evaluation / uncertainty estimation depending on the current QA tab.
        """
        self.evaluation_panel.clear_images_list()
        self.uncertainty_panel.clear_images_list()
        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            self.on_run_button(self.evaluation)
        else:
            self.on_run_button(self.uncertainty)

    def cleanup(self) -> None:
        super().cleanup()
        settings = QSettings()
        if len(self.app_local_repositoy) > 0:
            settings.setValue(
                f"KonfAI-Settings/{self._name}/Apps",
                json.dumps(self.app_local_repositoy),
            )

    def evaluation(self, remote_server: RemoteServer, devices: list[str]) -> None:
        """
        Run reference-based evaluation using the selected app.

        Steps:
          - Resample input volume and optional mask to the reference volume space
          - Export input and reference volumes (and mask if present) to .mha files
          - Call `konfai-apps eval` in the temporary work directory
          - Read resulting JSON metrics and MHA images and display them in the panel
        """
        self.evaluation_panel.clear_metrics()

        # Ensure we have a transform node; if not, create an identity transform
        if not self.ui.inputTransformSelector.currentNode():
            new_transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "IdentityTransform")
            self.ui.inputTransformSelector.setCurrentNode(new_transform)
            self.set_parameter_node("InputTransform", self.ui.inputTransformSelector.currentNodeID)

        # Create an output node to store the warped input volume (aligned to reference)
        output_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode",
            self.ui.inputVolumeEvaluationSelector.currentNode().GetName() + "_toRef",
        )

        input_volume_evaluation_node = self.ui.inputVolumeEvaluationSelector.currentNode()
        reference_volume_evaluation_node = self.ui.referenceVolumeSelector.currentNode()

        # Resample the input volume to match the reference grid, using the selected transform
        params = {
            "inputVolume": input_volume_evaluation_node.GetID(),
            "referenceVolume": reference_volume_evaluation_node.GetID(),
            "outputVolume": output_node.GetID(),
            "interpolationType": "linear",
            "warpTransform": self.ui.inputTransformSelector.currentNode().GetID(),
        }

        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)

        # Write the resampled volume and reference volume to disk
        volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volume_storage_node.SetFileName(str(self._work_dir / "Volume.mha"))
        volume_storage_node.UseCompressionOff()
        volume_storage_node.WriteData(output_node)
        volume_storage_node.UnRegister(None)

        volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volume_storage_node.SetFileName(str(self._work_dir / "Reference.mha"))
        volume_storage_node.UseCompressionOff()
        volume_storage_node.WriteData(reference_volume_evaluation_node)
        volume_storage_node.UnRegister(None)

        app = self.ui.appComboBox.currentData

        # Build konfai-apps evaluation CLI arguments
        args = [
            "eval",
            app.get_name(),
            "-i",
            "Volume.mha",
            "--gt",
            "Reference.mha",
            "-o",
            "Evaluation",
        ]

        # Optional: resample and include the reference mask if defined
        if has_node_content(self.ui.referenceMaskSelector.currentNode()):
            output_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode",
                self.ui.referenceMaskSelector.currentNode().GetName() + "_toRef",
            )
            params = {
                "inputVolume": self.ui.referenceMaskSelector.currentNode().GetID(),
                "referenceVolume": self.ui.referenceVolumeSelector.currentNode().GetID(),
                "outputVolume": output_node.GetID(),
                "interpolationType": "nn",
                "warpTransform": self.ui.inputTransformSelector.currentNode().GetID(),
            }

            slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)

            mask_storage = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            mask_storage.SetFileName(str(self._work_dir / "Mask.mha"))
            mask_storage.UseCompressionOff()
            mask_storage.WriteData(output_node)
            mask_storage.UnRegister(None)

            args += ["--mask", "Mask.mha"]

        # Select device backend from parameter node (GPU indices or CPU fallback)
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

        def on_end_function() -> None:
            """
            Callback executed when the evaluation process finishes.

            It loads metrics and images produced by konfai-apps and updates
            the evaluation panel accordingly.
            """
            if not any((self._work_dir / "Evaluation").rglob("*.json")):
                return

            # Read the first JSON metrics file found in the Evaluation directory
            from konfai.evaluator import Statistics

            statistics = Statistics(next((self._work_dir / "Evaluation").rglob("*.json")))
            self.evaluation_panel.set_metrics(statistics.read())

            # Populate the image list with all .mha images in the Evaluation folder
            self.evaluation_panel.refresh_images_list(Path(next((self._work_dir / "Evaluation").rglob("*.mha")).parent))

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)

    def uncertainty(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """
        Run uncertainty analysis using the selected app.

        Steps:
          - Export the inference stack from Slicer to an MHA file
          - Call `konfai-apps uncertainty` with the selected app
          - Read resulting JSON metrics and uncertainty maps and display them
        """
        # Clear previous metrics and images in both panels
        self.uncertainty_panel.clear_metrics()

        app = self.ui.appComboBox.currentData
        args = [
            "uncertainty",
            app.get_name(),
            "-i",
            "Volume.mha",
            "-o",
            "Uncertainty",
        ]

        # Device selection (GPU or CPU)
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

        n = self.ui.inputVolumeSequenceSelector.currentNode().GetNumberOfDataNodes()
        images = [
            sitkUtils.PullVolumeFromSlicer(self.ui.inputVolumeSequenceSelector.currentNode().GetNthDataNode(i))
            for i in range(n)
        ]
        arrays = [sitk.GetArrayFromImage(img) for img in images]
        stack = np.stack(arrays, axis=-1)
        image = sitk.GetImageFromArray(stack, isVector=True)
        image.CopyInformation(images[0])

        sitk.WriteImage(image, str(self._work_dir / "Volume.mha"))

        def on_end_function() -> None:
            """
            Callback executed when the uncertainty process finishes.

            It loads metrics and MHA images produced by konfai-apps and updates
            the uncertainty panel accordingly.
            """
            if not any((self._work_dir / "Uncertainty").rglob("*.json")):
                return

            from konfai.evaluator import Statistics

            statistics = Statistics(next((self._work_dir / "Uncertainty").rglob("*.json")))
            self.uncertainty_panel.set_metrics(statistics.read())
            self.uncertainty_panel.refresh_images_list(
                Path(next((self._work_dir / "Uncertainty").rglob("*.mha")).parent)
            )

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)

    def inference(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """
        Run inference using the selected KonfAI app.

        Steps:
          - Export the input volume to an MHA file with appropriate metadata
          - Call `konfai-apps infer` with ensemble/TTA/MC-dropout options
          - Load the resulting MHA files back into Slicer and update display
          - Populate uncertainty panel with the generated stack if available
        """

        self.evaluation_panel.clear_images_list()
        self.uncertainty_panel.clear_images_list()
        self.evaluation_panel.clear_metrics()
        self.uncertainty_panel.clear_metrics()

        app = self.ui.appComboBox.currentData
        args = [
            "infer",
            app.get_name(),
            "-i",
            "Volume.mha",
            "-o",
            "Output",
            "--ensemble_models",
            *self.chip_selector.selected(),
            "--tta",
            str(self.ui.ttaSpinBox.value),
            "--mc",
            str(self.ui.mcDropoutSpinBox.value),
        ]
        if self.ui.uncertaintyCheckBox.isChecked():
            args += ["-uncertainty"]

        # Device selection (GPU or CPU)
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

        input_node = self.ui.inputVolumeSelector.currentNode()

        # Export volume to disk using SimpleITK
        sitk_image = sitkUtils.PullVolumeFromSlicer(input_node)

        # Store KonfAI metadata in the MHA header
        sitk_image.SetMetaData(
            "App",
            app.get_name(),
        )
        sitk_image.SetMetaData("NumberOfEnsemble", f"{self.ui.ensembleSpinBox.value}")
        sitk_image.SetMetaData("NumberOfTTA", f"{self.ui.ttaSpinBox.value}")
        sitk_image.SetMetaData("NumberOfMCDropout", f"{self.ui.mcDropoutSpinBox.value}")

        sitk.WriteImage(sitk_image, str(self._work_dir / "Volume.mha"))

        self._update_logs(f"Input volume saved to temporary folder: {self._work_dir / 'Volume.mha'}")

        def on_end_function() -> None:
            """
            Callback executed when inference finishes.

            It reads the first non-stack output file, loads it into Slicer, sets
            appropriate volume class (scalar vs labelmap), copies orientation and
            attributes, and configures slice viewer overlays.
            """
            data = None
            # Find the first non-stack MHA output file
            for file in (self._work_dir / "Output").rglob("*.mha"):
                if file.name != "InferenceStack.mha":
                    from konfai.utils.dataset import image_to_data

                    data, attr = image_to_data(sitk.ReadImage(str(file)))
                    break
            if data is None:
                return

            self._update_logs("Loading result into Slicer...")

            # Decide if output should be a label map (uint8) or scalar volume
            want_label = data.dtype == np.uint8
            expected_class = "vtkMRMLSegmentationNode" if want_label else "vtkMRMLScalarVolumeNode"
            base_name = (
                self.ui.outputVolumeSelector.baseName
                + "_"
                + app.get_name()
            )

            node = slicer.mrmlScene.AddNewNodeByClass(expected_class, base_name+ ("_Segmentation" if want_label else "_Output"))
            self.ui.outputVolumeSelector.setCurrentNode(node)
            self.ui.segmentationShow3DButton.setVisible(node.IsA("vtkMRMLSegmentationNode"))
            if node.IsA("vtkMRMLSegmentationNode"):
                tmp_labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", base_name+"_LabelMap")
                slicer.util.updateVolumeFromArray(tmp_labelmap, data[0])
                tmp_labelmap.CopyOrientation(self.ui.inputVolumeSelector.currentNode())
                node.CreateDefaultDisplayNodes()

                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tmp_labelmap, node)
                label_value_to_segment_name = app.get_terminology()
                segmentation = node.GetSegmentation()
                for label_value, segment_id in zip(np.unique(data[0])[1:], range(segmentation.GetNumberOfSegments())):
                    segment = segmentation.GetNthSegment(segment_id)
                    
                    label_value = int(label_value)

                    if label_value in label_value_to_segment_name:
                        segment.SetName(label_value_to_segment_name[label_value].name)

                        def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
                            hex_color = hex_color.lstrip("#")
                            r = int(hex_color[0:2], 16) / 255.0
                            g = int(hex_color[2:4], 16) / 255.0
                            b = int(hex_color[4:6], 16) / 255.0
                            return r, g, b

                        r, g, b = hex_to_rgb01(label_value_to_segment_name[label_value].color)
                        segment.SetColor(r, g, b)
                    else:
                        random.seed(label_value)
                        segment.SetColor(random.random(), random.random(), random.random())
            else:
                # Populate the node with the first channel of the prediction data
                slicer.util.updateVolumeFromArray(node, data[0])

                # Copy orientation from the input volume so they are aligned
                node.CopyOrientation(self.ui.inputVolumeSelector.currentNode())

            # Copy KonfAI metadata to MRML node attributes
            for key, value in attr.items():
                node.SetAttribute(key.split("_")[0], str(value))

            # Configure slice viewer overlays depending on label vs scalar output
            if want_label:
                slicer.util.setSliceViewerLayers(
                    label=node,
                    background=self.ui.inputVolumeSelector.currentNode(),
                    fit=True,
                )
            else:
                slicer.util.setSliceViewerLayers(
                    foreground=self.ui.outputVolumeSelector.currentNode(),
                    background=self.ui.inputVolumeSelector.currentNode(),
                    fit=True,
                    foregroundOpacity=0.5,
                )

            sequence_node = self.ui.inputVolumeSequenceSelector.currentNode()
            if sequence_node is None:
                sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "InputTransformSequence")
                self.ui.inputVolumeSequenceSelector.setCurrentNode(sequence_node)
            else:
                sequence_node.RemoveAllDataNodes()

            for file in (self._work_dir / "Output").rglob("*.mha"):
                if file.name == "InferenceStack.mha":
                    inference_stack = sitk.ReadImage(str(file))
                    if inference_stack.GetNumberOfComponentsPerPixel() > 1:
                        data = sitk.GetArrayFromImage(inference_stack)
                        for i in range(data.shape[-1]):
                            img = sitk.GetImageFromArray(data[..., i])
                            img.CopyInformation(inference_stack)
                            temp_volume_node = sitkUtils.PushVolumeToSlicer(img, name=f"Output_stack_{i}")
                            sequence_node.SetDataNodeAtValue(temp_volume_node, str(i))
                            slicer.mrmlScene.RemoveNode(temp_volume_node)

                    browser_node = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLSequenceBrowserNode", "SitkTransformSequenceBrowser"
                    )
                    browser_node.SetAndObserveMasterSequenceNodeID(sequence_node.GetID())
                    for key, value in attr.items():
                        sequence_node.SetAttribute(key.split("_")[0], str(value))
                    break

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)


class RemoteServerConfigDialog(QDialog):

    def __init__(self, remote_server: RemoteServer):
        super().__init__()
        self.remote_server = remote_server
        self.setWindowTitle(f"Configure server: {remote_server.name}")
        self.setModal(True)

        self._remove = False  # <-- flag

        self.hostEdit = QLineEdit(remote_server.host)
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(int(remote_server.port))
        self.tokenEdit = QLineEdit(remote_server.token)
        self.tokenEdit.setPlaceholderText("Optional (Bearer token)")
        self.tokenEdit.setEchoMode(QLineEdit.EchoMode.Password)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)

        self.checkButton = QPushButton("Check")
        self.saveButton = QPushButton("Save")
        self.removeButton = QPushButton("Remove")
        self.cancelButton = QPushButton("Cancel")

        form = QFormLayout()
        form.addRow("Host", self.hostEdit)
        form.addRow("Port", self.portSpin)
        form.addRow("Token", self.tokenEdit)

        btns = QHBoxLayout()
        btns.addWidget(self.checkButton)
        btns.addStretch(1)
        btns.addWidget(self.removeButton)
        btns.addWidget(self.saveButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.statusLabel)
        root.addLayout(btns)

        self.cancelButton.clicked.connect(lambda _=False: self.reject())
        self.saveButton.clicked.connect(lambda _=False: self.on_save())
        self.removeButton.clicked.connect(lambda _=False: self.on_remove())
        self.checkButton.clicked.connect(lambda _=False: self.on_check())

    def on_check(self) -> bool:
        self.remote_server.host = self.hostEdit.text.strip()
        self.remote_server.port = int(self.portSpin.value)
        self.remote_server.token = self.tokenEdit.text.strip()
        from konfai import check_server

        ok, msg = check_server(self.remote_server)
        self.statusLabel.setText(("✅ " if ok else "❌ ") + msg)
        return ok

    def on_remove(self):
        self._remove = True
        self.accept()

    def on_save(self):
        if self.on_check():
            self.accept()

    def get(self) -> RemoteServer:
        self.remote_server.host = self.hostEdit.text.strip()
        self.remote_server.port = int(self.portSpin.value)
        self.remote_server.token = self.tokenEdit.text.strip()
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring

        keyring.set_password(SERVICE, str(self.remote_server), self.remote_server.token)
        return self.remote_server

    def want_remove(self) -> bool:
        return self._remove


class RemoteServerAddDialog(QDialog):
    def __init__(self, remote_servers_name: list[str]):
        super().__init__()
        self.remote_servers_name = remote_servers_name
        self.setWindowTitle("Add remote server")
        self.setModal(True)

        self.nameEdit = QLineEdit("")
        self.hostEdit = QLineEdit("127.0.0.1")
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(8000)
        self.tokenEdit = QLineEdit("")
        self.tokenEdit.setPlaceholderText("Optional (Bearer token)")
        self.tokenEdit.setEchoMode(QLineEdit.EchoMode.Password)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)

        self.checkButton = QPushButton("Check")
        self.addButton = QPushButton("Add")
        self.cancelButton = QPushButton("Cancel")

        form = QFormLayout()
        form.addRow("Name", self.nameEdit)
        form.addRow("Host", self.hostEdit)
        form.addRow("Port", self.portSpin)
        form.addRow("Token", self.tokenEdit)

        btns = QHBoxLayout()
        btns.addWidget(self.checkButton)
        btns.addStretch(1)
        btns.addWidget(self.addButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.statusLabel)
        root.addLayout(btns)

        self.checkButton.clicked.connect(self.on_check)
        self.addButton.clicked.connect(self.on_add)
        self.cancelButton.clicked.connect(lambda _=False: self.reject())

    def on_check(self) -> bool:
        remote_server = self.get()
        from konfai import check_server

        ok, msg = check_server(remote_server)
        self.statusLabel.setText(("✅ " if ok else "❌ ") + msg)
        return ok

    def on_add(self):
        name = self.nameEdit.text.strip()
        if not name:
            self.statusLabel.setText("❌ Name is required")
            return
        if name in self.remote_servers_name:
            self.statusLabel.setText("❌ This server name already exists.")
            return
        if self.on_check():
            self.accept()

    def get(self) -> RemoteServer:
        name = self.nameEdit.text.strip()
        host = self.hostEdit.text.strip()
        port = int(self.portSpin.value)

        token = self.tokenEdit.text.strip() or None
        id = f"{name}|{host}|{port}"
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring

        if token:
            keyring.set_password(SERVICE, id, token)
        else:
            try:
                keyring.delete_password(SERVICE, id)
            except keyring.errors.PasswordDeleteError:
                pass

        return RemoteServer(name, host, port)


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
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.initialize_parameter_node()
        # Set header title
        self.ui.headerTitleLabel.text = title

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

        self.ui.deviceComboBox.currentIndexChanged.connect(self.on_device_changed)
        self.ui.remoteServerComboBox.setStyleSheet(
            """
            QComboBox[status="ok"]  { color: rgb(0,160,0); }
            QComboBox[status="bad"] { color: rgb(200,0,0); }
            """
        )

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
            app.app_setup(self.update_logs, self.update_progress, self._parameter_node)

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
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_konfai_app.get_work_dir()))

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

        if used_gb / total_gb * 100 > 80:
            # Red when RAM usage is high
            self.ui.ramProgressBar.setStyleSheet(
                """
                QProgressBar::chunk {
                    background-color: #e74c3c;
                }
            """
            )
        else:
            # Green otherwise
            self.ui.ramProgressBar.setStyleSheet(
                """
                QProgressBar::chunk {
                    background-color: #2ecc71; 
                }
            """  # noqa: W291
            )

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

            if used_gb / total_gb * 100 > 80:
                self.ui.gpuProgressBar.setStyleSheet(
                    """
                    QProgressBar::chunk {
                        background-color: #e74c3c;
                    }
                """
                )
            else:
                # Green otherwise
                self.ui.gpuProgressBar.setStyleSheet(
                    """
                    QProgressBar::chunk {
                        background-color: #2ecc71;
                    }
                """
                )
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
        self._update_ram()
        self._update_vram()
        if clear:
            self.ui.logText.plainText = text
        else:
            self.ui.logText.appendPlainText(text)

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
            raw = settings.value(f"KonfAI-Settings/RemoteServers")
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

    def exit(self) -> None:
        if self._current_konfai_app:
            self._current_konfai_app.exit()


def _is_reload_setup(moduleName: str) -> bool:
    key = f"{moduleName}.wasSetupOnce"
    was = bool(slicer.app.property(key))
    slicer.app.setProperty(key, True)
    return was


class KonfAIWidget(ScriptedLoadableModuleWidget):
    """
    Top-level scripted loadable module widget for KonfAI.

    This class ties together the Slicer module system with the KonfAICoreWidget,
    which handles actual application logic and GUI.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().__init__(parent)

    def setup(self) -> None:
        """
        Construct and initialize the KonfAI module GUI.

        This method is called once when the user first opens the module.
        """
        super().setup()
        # Create the core KonfAI widget
        self.konfai_core = KonfAICoreWidget("KonfAI Apps")

        # Create and register one KonfAI app specialized for inference
        prediction_widget = KonfAIAppTemplateWidget(
            "Inference",
            ["VBoussot/ImpactSynth", "VBoussot/MRSegmentator-KonfAI", "VBoussot/TotalSegmentator-KonfAI"],
        )
        self.konfai_core.register_apps([prediction_widget])

        # Attach the core widget to the Slicer module layout
        self.layout.addWidget(self.konfai_core)

        if _is_reload_setup("SlicerKonfAI"):
            self.konfai_core.enter()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.

        This hook can be used to ensure state is up-to-date when the user
        returns to the module. Currently no additional logic is required.
        """
        self.konfai_core.enter()

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.

        This hook can be used to pause or finalize ongoing tasks, but
        no special handling is required at the moment.
        """
        self.konfai_core.exit()
