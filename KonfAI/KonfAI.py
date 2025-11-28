import slicer

try:
    from konfai.evaluator import Statistics
except ImportError:
    # Install KonfAI inside Slicer if it is not available yet
    slicer.util.pip_install("konfai==1.4.2")
    from konfai.evaluator import Statistics

import itertools
import json
import os
import re
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pynvml
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import vtk
from konfai.utils.dataset import get_infos, image_to_data
from konfai.utils.utils import (
    AppDirectoryError,
    AppRepositoryHFError,
    ModelDirectory,
    ModelHF,
    get_available_models_on_hf_repo,
)
from qt import (
    QCheckBox,
    QCursor,
    QDesktopServices,
    QFileDialog,
    QFont,
    QIcon,
    QInputDialog,
    QLineEdit,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProcess,
    QSettings,
    QSize,
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
from torch.cuda import device_count, get_device_name, is_available


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
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Deep Learning")]
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

    def __init__(self, _update_logs, _update_progress):
        super().__init__(self)
        self.readyReadStandardOutput.connect(self.on_stdout_ready)
        self.readyReadStandardError.connect(self.on_stderr_ready)

        # Callbacks defined by the KonfAI core widget
        self._update_logs = _update_logs
        self._update_progress = _update_progress

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
            self._update_logs(line)

            # Parse progress percentage if present (e.g., " 45% 123/456")
            m = re.search(r"\b(\d{1,3})%(?=\s+\d+/\d+)", line)
            # Parse speed pattern if present (e.g., "5.2 it/s" or "0.21 s/it")
            speed = re.search(r"([\d.]+)\s*(it/s|s/it)", line)

            if m:
                pct = int(m.group(1))
            else:
                pct = None

            if speed and pct is not None:
                # Notify UI of updated progress and speed
                self._update_progress(pct, speed.group(1) + " " + speed.group(2))

    def on_stderr_ready(self) -> None:
        """
        Handle new data available on the standard error stream.

        Currently we simply print the content to the Python console for debugging.
        """
        print("Error : ", self.readAllStandardError().data().decode().strip())

    def run(self, command: str, work_dir: Path, args: list[str], on_end_function) -> None:
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

        # Disconnect any previous 'finished' slot to avoid stacking connections
        try:
            self.finished.disconnect()
        except TypeError:
            # No slot connected yet
            pass

        self.finished.connect(on_end_function)

        # Start the process asynchronously
        self.start(command, args)

    def stop(self) -> None:
        """
        Immediately terminate the running process, if any.
        """
        self.kill()
        self.waitForFinished(-1)


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
        self.process = Process(update_logs, update_progress)

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
        if self._parameter_node is not None:
            self._parameter_node.SetParameter(f"{self._name}/{key}", str(value))

    def set_parameter_node(self, key: str, value) -> None:
        if self._parameter_node is not None:
            self._parameter_node.SetNodeReferenceID(f"{self._name}/{key}", value)

    def get_parameter(self, key: str) -> str | bool:
        if self._parameter_node is not None:
            return self._parameter_node.GetParameter(f"{self._name}/{key}")
        else:
            return False

    def get_parameter_node(self, key: str):
        if self._parameter_node is not None:
            return self._parameter_node.GetNodeReference(f"{self._name}/{key}")
        else:
            return None

    def get_device(self) -> str | None:
        if self._parameter_node is not None and self._parameter_node.GetParameter("Device") != "None":
            return self._parameter_node.GetParameter("Device")
        else:
            return None

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
            self.remove_work_dir()
            self.create_new_work_dir()
            self.set_running(True)
            self._update_logs("Processing started.", True)
            self._update_progress(0, "0 it/s")
            try:
                function()
            except Exception as e:
                # Log the exception for debugging and reset running state
                print(e)
                self.set_running(False)
        else:
            # Stop current operation
            self.set_running(False)
            self.process.stop()
            # Give the process some time to terminate properly
            import time

            time.sleep(3)

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
        self.initialize_gui_from_parameter_node()
        self.update_gui_from_parameter_node()


class KonfAIAppTemplateWidget(AppTemplateWidget):
    """
    Concrete implementation of AppTemplateWidget for KonfAI applications.

    This widget provides:
      - Input/output volume selection for inference
      - QA capabilities (evaluation with reference, or uncertainty estimation)
      - Model selection from Hugging Face repositories
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

        settings = QSettings()
        raw = settings.value(f"KonfAI-Settings/{name}/Models")
        models_name = []
        if raw is not None:
            models_name = json.loads(raw)

        default_models_name = []
        for konfai_repo in konfai_repo_list:
            try:
                default_models_name += [
                    konfai_repo + ":" + model_name for model_name in get_available_models_on_hf_repo(konfai_repo)
                ]
            except AppRepositoryHFError as e:
                slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
        models_name = list(set(default_models_name + models_name))
        # Populate the model combo box with models found in the provided Hugging Face repos
        for model_name in models_name:
            try:
                if len(model_name.split(":")) == 2:
                    model = ModelHF(model_name.split(":")[0], model_name.split(":")[1])
                else:
                    model = ModelDirectory(Path(model_name).parent, Path(model_name).name)
                self.ui.modelComboBox.addItem(model.get_display_name(), model)
            except Exception as e:
                slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")

        self.ui.modelComboBox.setCurrentIndex(0)

        # Connect volume selectors to parameter node synchronization
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        # Evaluation / uncertainty input selectors
        self.ui.inputVolumeEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.on_input_volume_evaluation_changed
        )
        self.ui.inputVolumeSequenceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.on_input_volume_evaluation_changed
        )

        # Reference and transform nodes
        self.ui.referenceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.referenceMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.inputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        self.ui.ensembleSpinBox.valueChanged.connect(self.on_ensemble_changed)
        self.ui.ttaSpinBox.valueChanged.connect(self.on_tta_changed)
        self.ui.mcDropoutSpinBox.valueChanged.connect(self.on_mc_dropout_changed)

        # Model selection and management
        self.ui.modelComboBox.currentIndexChanged.connect(self.on_model_selected)
        self.ui.addModelButton.clicked.connect(self.on_add_model)
        self.ui.removeModelButton.clicked.connect(self.on_remove_model)

        # Configuration button (opens folder containing KonfAI YAML configs)
        icon_path = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "gear.png")
        self.ui.configButton.setIcon(QIcon(icon_path))
        self.ui.configButton.setIconSize(QSize(18, 18))
        self.ui.configButton.clicked.connect(self.on_open_config)

        # Run buttons for inference and QA
        self.ui.runInferenceButton.clicked.connect(self.on_run_inference_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        # Description toggle and QA tab changes
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)

    def on_ensemble_changed(self):
        self.set_parameter("number_of_ensemble", str(self.ui.ensembleSpinBox.value))

    def on_tta_changed(self):
        self.set_parameter("number_of_tta", str(self.ui.ttaSpinBox.value))

    def on_mc_dropout_changed(self):
        self.set_parameter("number_of_mc_dropout", str(self.ui.mcDropoutSpinBox.value))

    def set_information(
        self,
        model: str | None = None,
        number_of_ensemble: int | None = None,
        number_of_tta: int | None = None,
        number_of_mc_dropout: int | None = None,
    ) -> None:
        """
        Update the model information summary panel (model name + ensemble/TTA/MC counts).

        When any field is None or not available, a placeholder is displayed.
        """
        self.ui.modelSummaryValue.setText(f"Model: {model}" if model else "Model: N/A")
        self.ui.ensembleSummaryValue.setText(f"#{number_of_ensemble}" if number_of_ensemble else "#N/A")
        self.ui.ttaSummaryValue.setText(f"#{number_of_tta}" if number_of_ensemble and number_of_tta else "#N/A")
        self.ui.mcSummaryValue.setText(
            f"#{number_of_mc_dropout}" if number_of_ensemble and number_of_mc_dropout else "#N/A"
        )

    def on_input_volume_evaluation_changed(self, node) -> None:
        """
        Handler called when the evaluation input or stack input selection changes.

        It attempts to read metadata from the selected node (or its storage),
        updates the model information summary, and synchronizes the parameter node.
        """
        if node:
            storage = node.GetStorageNode()
            if storage:
                path = storage.GetFileName()
                if path and Path(path).exists():
                    _, attr = get_infos(path)
                    if (
                        "Model" in attr
                        and "NumberOfEnsemble" in attr
                        and "NumberOfTTA" in attr
                        and "NumberOfMCDropout" in attr
                    ):
                        self.set_information(
                            attr["Model"],
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
                    node.GetAttribute("Model"),
                    node.GetAttribute("NumberOfEnsemble"),
                    node.GetAttribute("NumberOfTTA"),
                    node.GetAttribute("NumberOfMCDropout"),
                )
        self.update_parameter_node_from_gui()

    def enter(self) -> None:
        """
        Overridden AppTemplateWidget entry point.

        Re-initializes parameter node, GUI state and ensures model selection
        is consistent when the widget is shown.
        """
        super().enter()
        self.on_model_selected()

    def initialize_parameter_node(self) -> None:
        """
        Initialize the parameter node with default values for this app
        (input volume, model ID, ensemble/TTA/MC-dropout parameters).
        """
        self._initialized = False

        # Select default input nodes if nothing is selected yet
        if self.get_parameter_node("InputVolume") is None:
            first_volume_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if first_volume_node and self._parameter_node is not None:
                self._parameter_node.SetNodeReferenceID(f"{self._name}/InputVolume", first_volume_node.GetID())

        # Set default model if none is stored yet
        if not self.get_parameter("Model"):
            model: ModelHF = self.ui.modelComboBox.itemData(0)
            self.set_parameter("Model", model.get_name())

        # Determine the current model object from the stored parameter if possible
        current_model = None
        for i in range(self.ui.modelComboBox.count):
            model_tmp: ModelHF = self.ui.modelComboBox.itemData(i)
            if model_tmp.get_name() == self.get_parameter("Model"):
                current_model = model_tmp
                break
        if not current_model:
            current_model = self.ui.modelComboBox.itemData(0)

        # Ensemble / TTA / MC-dropout defaults based on model capabilities
        if not self.get_parameter("number_of_ensemble"):
            self.set_parameter("number_of_ensemble", str(current_model.get_number_of_models()))
        if not self.get_parameter("number_of_tta"):
            self.set_parameter("number_of_tta", str(current_model.get_maximum_tta()))
        if not self.get_parameter("number_of_mc_dropout"):
            self.set_parameter("number_of_mc_dropout", str(current_model.get_mc_dropout()))
        self.initialize_gui_from_parameter_node()
        self._initialized = True

    def initialize_gui_from_parameter_node(self) -> None:
        """
        Initialize GUI widget values from the parameter node.
        """
        # Model selection
        model_param = self.get_parameter("Model")
        # Search the combo box items for a matching model name
        index = -1
        for i in range(self.ui.modelComboBox.count):
            model: ModelHF = self.ui.modelComboBox.itemData(i)
            if model.get_name() == model_param:
                index = i
                break
        self.ui.modelComboBox.setCurrentIndex(index if index != -1 else 0)

        # Ensemble / TTA / MC-dropout spin boxes
        self.ui.ensembleSpinBox.setValue(int(self.get_parameter("number_of_ensemble")))
        self.ui.ttaSpinBox.setValue(int(self.get_parameter("number_of_tta")))
        self.ui.mcDropoutSpinBox.setValue(int(self.get_parameter("number_of_mc_dropout")))

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
        """
        Update the GUI state based on the current parameter node values.

        This includes enabling/disabling buttons, updating tooltips and
        configuring default output volume names.
        """
        input_volume = self.get_parameter_node("InputVolume")
        if input_volume and self.ui.modelComboBox.currentData:
            self.ui.runInferenceButton.toolTip = _("Start inference")
            self.ui.runInferenceButton.enabled = True
        else:
            self.ui.runInferenceButton.toolTip = _("Select input volume")
            self.ui.runInferenceButton.enabled = False

        # Update run/stop label based on running state
        if not self.is_running():
            self.ui.runInferenceButton.text = "Run"
            self.ui.runEvaluationButton.text = "Run"
        else:
            self.ui.runInferenceButton.text = "Stop"
            self.ui.runEvaluationButton.text = "Stop"

        reference_volume = self.get_parameter_node("ReferenceVolume")
        input_evaluation_volume = self.get_parameter_node("InputVolumeEvaluation")
        inference_stack_volume = self.get_parameter_node("InputVolumeSequence")

        # Configure QA button depending on which tab is active
        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            # Reference-based evaluation: need both input and reference volumes
            if (
                reference_volume
                and reference_volume.GetImageData()
                and input_evaluation_volume
                and input_evaluation_volume.GetImageData()
            ):
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
        (volume selectors, model selection, etc.).
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

    def on_model_selected(self) -> None:
        """
        Handle model selection changes.

        This method:
          - Resets description expansion
          - Adjusts ensemble / TTA / MC-dropout spin box ranges
          - Enables/disables QA options depending on model capabilities
        """
        model: ModelHF = self.ui.modelComboBox.currentData
        if model is None:
            return

        # Removing model from disk is only relevant for custom (local) models;
        # by default we disable the button for HF models.
        self.ui.removeModelButton.setEnabled(model.get_name().split(":")[0] not in self._konfai_repo_list)

        # Reset description expansion state and update description label
        self._description_expanded = False
        self.on_toggle_description()

        # Update ensemble / TTA / MC-dropout limits
        if int(self.get_parameter("number_of_ensemble")) > model.get_number_of_models():
            self.set_parameter("number_of_mc_dropout", str(model.get_number_of_models()))

        if int(self.get_parameter("number_of_tta")) > model.get_maximum_tta():
            self.set_parameter("number_of_tta", str(model.get_maximum_tta()))
        if int(self.get_parameter("number_of_mc_dropout")) > model.get_mc_dropout():
            self.set_parameter("number_of_mc_dropout", str(model.get_mc_dropout()))

        self.ui.ensembleSpinBox.setMaximum(model.get_number_of_models())

        self.ui.ttaSpinBox.setEnabled(model.get_maximum_tta() > 0)
        self.ui.ttaSpinBox.setMaximum(model.get_maximum_tta())

        self.ui.mcDropoutSpinBox.setEnabled(model._mc_dropout > 0)
        self.ui.mcDropoutSpinBox.setMaximum(model._mc_dropout)

        # Enable QA sections based on model capabilities (evaluation/uncertainty support)
        has_evaluation, has_uncertainty = model.has_capabilities()
        self.ui.evaluationCollapsible.setEnabled(has_evaluation or has_uncertainty)
        if not has_evaluation and not has_uncertainty:
            self.ui.evaluationCollapsible.collapsed = True

        # Enable/disable tabs
        self.ui.qaTabWidget.setTabEnabled(0, has_evaluation)
        self.ui.qaTabWidget.setTabEnabled(1, has_uncertainty)

        self.set_parameter("Model", model.get_name())

    def on_toggle_description(self) -> None:
        """
        Toggle between short and full model description text in the UI.
        """
        model: ModelHF = self.ui.modelComboBox.currentData
        if not model:
            return

        if self._description_expanded:
            self.ui.modelDescriptionLabel.setText(model.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.modelDescriptionLabel.setText(model.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")

        self._description_expanded = not self._description_expanded

    def on_remove_model(self) -> None:
        """
        Remove the currently selected model from the list, optionally deleting
        the corresponding directory from disk.

        This is intended for locally added models, not Hugging Face models.
        """
        model: ModelHF = self.ui.modelComboBox.currentData

        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("Remove model?")
        mb.setText(f"Do you really want to remove “{model.get_display_name()}” from the list?")
        mb.setInformativeText("This will remove the model entry from the extension’s list.")
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)

        chk = QCheckBox("Also delete the folder from disk")
        chk.setChecked(False)
        mb.setCheckBox(chk)

        if mb.exec_() != QMessageBox.Yes:
            return

        index = self.ui.modelComboBox.findText(model.get_display_name())
        self.ui.modelComboBox.removeItem(index)

        if chk.isChecked():
            if isinstance(model, ModelDirectory):
                try:
                    shutil.rmtree(model.get_name())
                    QMessageBox.information(
                        None, "Folder deleted", f"The folder has been successfully deleted:\n{model.get_name()}"
                    )
                except Exception as e:
                    QMessageBox.critical(None, "Deletion error", f"Failed to delete folder:\n{model.get_name()}\n\n{e}")

    def on_open_config(self) -> None:
        """
        Open the directory containing the inference configuration

        The folder is opened in the operating system's default file browser.
        """
        model: ModelHF = self.ui.modelComboBox.currentData
        if model is None:
            return

        _, inference_file_path, _ = model.download_inference(0)
        QDesktopServices.openUrl(QUrl.fromLocalFile(Path(inference_file_path).parent))

    def on_add_model(self) -> None:
        """
        Show a menu to add a new model (from a folder, from Hugging Face, or configure fine-tuning).
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
        Add a model located in a local folder to the model list.

        The folder is expected to contain a valid KonfAI model configuration (YAML + weights).
        """
        model_dir = QFileDialog.getExistingDirectory(None, "Select Model Folder", os.path.expanduser("~"))
        if not model_dir:
            return
        try:
            model = ModelDirectory(Path(model_dir).parent, Path(model_dir).name)
        except AppDirectoryError as e:
            slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
            return

        # Ensure uniqueness of display names to avoid confusion in the combo box
        items = [self.ui.modelComboBox.itemText(i) for i in range(self.ui.modelComboBox.count)]
        if model.get_display_name() in items:
            QMessageBox.critical(
                None,
                "Model already listed",
                f'The model "{model.get_display_name()}" is already in the list.',
            )
            return
        self.ui.modelComboBox.addItem(model.get_display_name(), model)
        self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.findData(model))

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
        Add a model from a Hugging Face repository.

        The user is prompted for a repo id (e.g., 'VBoussot/ImpactSynth') and then
        a subdirectory (model name) is selected among the available directories.
        """
        from huggingface_hub import HfApi

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

        # List top-level directories as candidate model names
        dirs = sorted({path.split("/")[0] for path in files if "/" in path})
        model_name = self.ask_subdir(dirs)
        if model_name:
            from konfai.utils.utils import is_model_repo

            state, error, _ = is_model_repo(repo_id, model_name)
            if not state:
                QMessageBox.critical(None, "Model", error)
                return

            model = ModelHF(repo_id, model_name)

            # Do not add duplicate display names
            items = [self.ui.modelComboBox.itemText(i) for i in range(self.ui.modelComboBox.count)]
            if model.get_display_name() in items:
                QMessageBox.critical(
                    None,
                    "Model already listed",
                    f'The model "{model.get_display_name()}" is already in the list.',
                )
                return

            self.ui.modelComboBox.addItem(model.get_display_name(), model)
            self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.findData(model))

    def on_add_ft(self) -> None:
        """
        Create a new folder intended for fine-tuning a model and add it to the list.

        This helper guides the user through selecting a parent directory,
        naming the fine-tune model folder and creating a basic skeleton with a README.
        """
        model: ModelHF = self.ui.modelComboBox.currentData
        if model is None:
            return

        # Choose the parent directory for the new model
        parent_dir = QFileDialog.getExistingDirectory(None, "Choose parent directory for the new model")
        if not parent_dir:
            return

        # Ask for the new model name
        name = QInputDialog.getText(None, "Fine tune", "New model name (folder will be created):")
        if not name.strip():
            return

        # Ask for the new model diplay name
        display_name = QInputDialog.getText(None, "Fine tune", "New diplay model name:")
        if not display_name.strip():
            return

        items = [self.ui.modelComboBox.itemText(i) for i in range(self.ui.modelComboBox.count)]
        if display_name in items:
            QMessageBox.critical(
                None,
                "Model already listed",
                f'The model "{display_name}" is already in the list.',
            )
            return

        # Make a filesystem-safe name (slug)
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        if not safe:
            QMessageBox.warning(None, "Invalid name", "Please enter a valid name.")
            return

        model.install_fine_tune(Path(parent_dir) / name, display_name)
        # Add the folder to the model combo box
        model_ft = ModelDirectory(Path(parent_dir), name)
        self.ui.modelComboBox.addItem(model_ft.get_display_name(), model_ft)
        self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.findData(model_ft))

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
        settings.setValue(
            f"KonfAI-Settings/{self._name}/Models",
            json.dumps([self.ui.modelComboBox.itemData(i).get_name() for i in range(self.ui.modelComboBox.count)]),
        )

    def evaluation(self) -> None:
        """
        Run reference-based evaluation using the selected model.

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

        # Resample the input volume to match the reference grid, using the selected transform
        params = {
            "inputVolume": self.ui.inputVolumeEvaluationSelector.currentNode().GetID(),
            "referenceVolume": self.ui.referenceVolumeSelector.currentNode().GetID(),
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
        volume_storage_node.WriteData(self.ui.referenceVolumeSelector.currentNode())
        volume_storage_node.UnRegister(None)

        model: ModelHF = self.ui.modelComboBox.currentData

        # Build konfai-apps evaluation CLI arguments
        args = [
            "eval",
            model.get_name(),
            "-i",
            "Volume.mha",
            "--gt",
            "Reference.mha",
            "-o",
            "Evaluation",
        ]

        # Optional: resample and include the reference mask if defined
        if self.ui.referenceMaskSelector.currentNode() and self.ui.referenceMaskSelector.currentNode().GetImageData():
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
        if self.get_device():
            args += ["--gpu", self.get_device()]
        else:
            args += ["--cpu", "1"]

        def on_end_function() -> None:
            """
            Callback executed when the evaluation process finishes.

            It loads metrics and images produced by konfai-apps and updates
            the evaluation panel accordingly.
            """
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            if not any((self._work_dir / "Evaluation").rglob("*.json")):
                self.set_running(False)
                return

            # Read the first JSON metrics file found in the Evaluation directory
            statistics = Statistics(next((self._work_dir / "Evaluation").rglob("*.json")))
            self.evaluation_panel.set_metrics(statistics.read())

            # Populate the image list with all .mha images in the Evaluation folder
            self.evaluation_panel.refresh_images_list(Path(next((self._work_dir / "Evaluation").rglob("*.mha")).parent))
            self._update_logs("Processing finished.")
            self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)

    def uncertainty(self) -> None:
        """
        Run uncertainty analysis using the selected model.

        Steps:
          - Export the inference stack from Slicer to an MHA file
          - Call `konfai-apps uncertainty` with the selected model
          - Read resulting JSON metrics and uncertainty maps and display them
        """
        # Clear previous metrics and images in both panels
        self.uncertainty_panel.clear_metrics()

        model: ModelHF = self.ui.modelComboBox.currentData
        args = [
            "uncertainty",
            model.get_name(),
            "-i",
            "Volume.mha",
            "-o",
            "Uncertainty",
        ]

        # Device selection (GPU or CPU)
        if self.get_device():
            args += ["--gpu", self.get_device()]
        else:
            args += ["--cpu", "1"]

        n = self.ui.inputVolumeSequenceSelector.currentNode().GetNumberOfDataNodes()
        images = [
            sitkUtils.PullVolumeFromSlicer(self.ui.inputVolumeSequenceSelector.currentNode().GetNthDataNode(i))
            for i in range(n)
        ]
        arrays = [sitk.GetArrayFromImage(img) for img in images]
        stack = np.stack(arrays, axis=-1)
        image = sitk.GetImageFromArray(stack)
        image.CopyInformation(images[0])

        sitk.WriteImage(image, str(self._work_dir / "Volume.mha"))

        def on_end_function() -> None:
            """
            Callback executed when the uncertainty process finishes.

            It loads metrics and MHA images produced by konfai-apps and updates
            the uncertainty panel accordingly.
            """
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return

            if not any((self._work_dir / "Uncertainty").rglob("*.json")):
                self.set_running(False)

            statistics = Statistics(next((self._work_dir / "Uncertainty").rglob("*.json")))
            self.uncertainty_panel.set_metrics(statistics.read())
            self.uncertainty_panel.refresh_images_list(
                Path(next((self._work_dir / "Uncertainty").rglob("*.mha")).parent)
            )
            self._update_logs("Processing finished.")
            self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)

    def inference(self) -> None:
        """
        Run inference using the selected KonfAI model.

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

        model: ModelHF = self.ui.modelComboBox.currentData
        args = [
            "infer",
            model.get_name(),
            "-i",
            "Volume.mha",
            "-o",
            "Output",
            "--ensemble",
            str(self.ui.ensembleSpinBox.value),
            "--tta",
            str(self.ui.ttaSpinBox.value),
            "--mc",
            str(self.ui.mcDropoutSpinBox.value),
        ]

        # Device selection (GPU or CPU)
        if self.get_device():

            args += ["--gpu", self.get_device()]
        else:
            args += ["--cpu", "1"]
        input_node = self.ui.inputVolumeSelector.currentNode()

        # Export volume to disk using SimpleITK
        sitk_image = sitkUtils.PullVolumeFromSlicer(input_node)

        # Store KonfAI metadata in the MHA header
        sitk_image.SetMetaData(
            "Model",
            model.get_name(),
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
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return

            data = None
            # Find the first non-stack MHA output file
            for file in (self._work_dir / "Output").rglob("*.mha"):
                if file.name != "InferenceStack.mha":
                    data, attr = image_to_data(sitk.ReadImage(str(file)))
                    break
            if data is None:
                self.set_running(False)
                return

            self._update_logs("Loading result into Slicer...")

            # Decide if output should be a label map (uint8) or scalar volume
            want_label = data.dtype == np.uint8
            expected_class = "vtkMRMLLabelMapVolumeNode" if want_label else "vtkMRMLScalarVolumeNode"
            base_name = "OutputLabel" if want_label else "OutputVolume"

            current = self.ui.outputVolumeSelector.currentNode()

            # Ensure that the current node has the correct MRML class
            if current is None:
                node = slicer.mrmlScene.AddNewNodeByClass(expected_class, base_name)
                self.ui.outputVolumeSelector.setCurrentNode(node)
            elif not current.IsA(expected_class):
                old = current
                slicer.mrmlScene.RemoveNode(old)

                node = slicer.mrmlScene.AddNewNodeByClass(expected_class, base_name)
                self.ui.outputVolumeSelector.setCurrentNode(node)
            else:
                node = current

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
                    break

            self._update_logs("Processing finished.")
            self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)


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

        # Populate GPU/CPU device combo box
        available_devices = self._get_available_devices()
        for available_device in available_devices:
            self.ui.deviceComboBox.addItem(available_device[0], available_device[1])

        self.ui.deviceComboBox.currentIndexChanged.connect(self.on_device_changed)
        # Default to the last device entry (typically a GPU combo if available)
        self.ui.deviceComboBox.setCurrentIndex(len(available_devices) - 1)
        # Observe scene close/open to keep parameter node in sync
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.on_scene_start_close)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.on_scene_end_close)

        # Work directory button configuration
        self.ui.openTempButton.setIcon(QIcon(resource_path("Icons/folder.png")))
        self.ui.openTempButton.setIconSize(QSize(18, 18))
        self.ui.openTempButton.clicked.connect(self.on_open_work_dir)
        self.ui.openTempButton.setEnabled(False)

        # Initialize log display
        self.update_logs("", True)

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
            app.initialize_parameter_node()

        # Enter the first app by default
        app = next(iter(self._apps.values()))
        self._current_konfai_app = app
        app.enter()

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
            self.on_device_changed()
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

    def _get_available_devices(self) -> list[tuple[str, str | None]]:
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
        available_devices: list[tuple[str, str | None]] = [("cpu [slow]", None)]

        if is_available():
            # Build combinations of GPU indices, so multi-GPU usage can be exposed
            combos: list[Any] = []
            nb_gpu = device_count()
            for r in range(1, nb_gpu + 1):
                combos.extend(itertools.combinations(range(nb_gpu), r))
            for device in combos:
                device_name = get_device_name(device[0])
                index = str(device[0])
                for i in device[1:]:
                    device_name += f",{get_device_name(i)}"
                    index += f"-{i}"
                available_devices.append((f"gpu {index} - {device_name}", index))
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
                self.ui.deviceComboBox.currentData if self.ui.deviceComboBox.currentData else "None",
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
        ram = psutil.virtual_memory()
        used_gb = (ram.total - ram.available) / (1024**3)
        total_gb = ram.total / (1024**3)
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
            try:
                used_gb = 0.0
                total_gb = 0.0
                pynvml.nvmlInit()
                for index in device.split(","):
                    info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(int(index)))
                    used_gb += info.used / (1024**3)
                    total_gb += info.total / (1024**3)

                self.ui.gpuLabel.show()
                self.ui.gpuProgressBar.show()
                self.ui.gpuLabel.text = _("VRAM used: {used:.1f} GB / {total:.1f} GB").format(
                    used=used_gb, total=total_gb
                )
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

            except Exception:
                # If NVML or GPU query fails, VRAM usage is reported as not available
                self.ui.gpuLabel.text = _("VRAM used: n/a")
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

    def update_progress(self, value: int, speed: float) -> None:
        """
        Update the progress bar and speed label.

        Parameters
        ----------
        value : int
            Progress percentage (0–100).
        speed : float | str
            Human-readable speed information, e.g. "5.2 it/s".
        """
        self._update_ram()
        self._update_vram()
        self.ui.progressBar.value = value
        self.ui.speedLabel.text = _("{speed}").format(speed=speed)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.

        Delegates cleanup to each registered KonfAI app (e.g., to remove
        temporary working directories).
        """
        for app in self._apps.values():
            app.cleanup()


class KonfAIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        VTKObservationMixin.__init__(self)

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

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.

        This hook can be used to ensure state is up-to-date when the user
        returns to the module. Currently no additional logic is required.
        """
        pass

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.

        This hook can be used to pause or finalize ongoing tasks, but
        no special handling is required at the moment.
        """
        pass
