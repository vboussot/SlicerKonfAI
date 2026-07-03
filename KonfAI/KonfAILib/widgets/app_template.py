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

"""Application template widget: base class shared with sister extensions
and the KonfAI app implementation (selection, inference, QA panels)."""

import shutil
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import slicer
from qt import (
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from slicer.i18n import tr as _

from KonfAILib.logic.mrml import has_node_content
from KonfAILib.logic.process import Process
from KonfAILib.logic.servers import RemoteServer
from KonfAILib.widgets.helpers import resource_path
from KonfAILib.widgets.panels.inference import KonfAIAppInferencePanel
from KonfAILib.widgets.panels.qa import KonfAIAppQAPanel
from KonfAILib.widgets.panels.selection import KonfAIAppSelectionPanel


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

    def app_setup(
        self,
        update_logs,
        update_progress,
        parameter_node,
        begin_status_progress: Callable[[], None] | None = None,
        end_status_progress: Callable[[], None] | None = None,
    ) -> None:
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
        self._begin_status_progress = begin_status_progress
        self._end_status_progress = end_status_progress
        self.process = Process(update_logs, update_progress, self.set_running)

    def _set_status_progress(self, value: int, message: str) -> None:
        self._update_progress(value, message)
        slicer.app.processEvents()

    @contextmanager
    def transient_status_progress(self, initial_message: str):
        if getattr(self, "_begin_status_progress", None) is not None:
            self._begin_status_progress()
        try:
            self._set_status_progress(0, initial_message)
            yield
        finally:
            if getattr(self, "_end_status_progress", None) is not None:
                self._end_status_progress()

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

    def exit(self) -> None:  # noqa: A003
        pass

    @abstractmethod
    def on_remote_server_changed(self):
        raise NotImplementedError


class KonfAIAppTemplateWidget(AppTemplateWidget):
    """
    Concrete implementation of AppTemplateWidget for KonfAI applications.

    This widget provides:
      - Input/output volume selection for inference
      - QA capabilities (evaluation with reference, or uncertainty estimation)
      - App selection from Hugging Face repositories
      - Configuration access for KonfAI Apps (YAML files)
    """

    # Panel classes are generic extension points: a domain-specific module
    # (e.g. registration) may subclass a panel to adapt the CLI it drives and
    # substitute it here, reusing the rest of the widget unchanged.
    SELECTION_PANEL_CLASS: type[KonfAIAppSelectionPanel] = KonfAIAppSelectionPanel
    INFERENCE_PANEL_CLASS: type[KonfAIAppInferencePanel] = KonfAIAppInferencePanel
    QA_PANEL_CLASS: type[KonfAIAppQAPanel] = KonfAIAppQAPanel

    def __init__(self, name: str, konfai_repo_list: list[str]):
        super().__init__(name, slicer.util.loadUI(resource_path("UI/KonfAIAppTemplate.ui")))
        self._konfai_repo_list = konfai_repo_list

        # Child panels: each panel loads its own .ui and connects its own signals
        self.selection_panel = self.SELECTION_PANEL_CLASS(self)
        self.ui.appSelectionPlaceholder.layout().addWidget(self.selection_panel)
        self.inference_panel = self.INFERENCE_PANEL_CLASS(self)
        self.ui.inferenceCollapsible.layout().addWidget(self.inference_panel)
        self.qa_panel = self.QA_PANEL_CLASS(self)
        self.ui.evaluationCollapsible.layout().addWidget(self.qa_panel)
        self._panels = [self.selection_panel, self.inference_panel, self.qa_panel]

        # Recompose the ui attribute map as the union of the template .ui and
        # all injected panel .ui files (widget names are unique across panels),
        # so the synchronization methods kept on the template work unchanged.
        self.ui = slicer.util.childWidgetVariables(self)

    @property
    def app_local_repositoy(self) -> list[str]:
        """Compatibility accessor: the state lives on the selection panel."""
        return self.selection_panel.app_local_repositoy

    @property
    def chip_selector(self):
        """Compatibility accessor: the chip selector lives on the inference panel."""
        return self.inference_panel.chip_selector

    @property
    def evaluation_panel(self):
        """Compatibility accessor: the evaluation metrics panel lives on the QA panel."""
        return self.qa_panel.evaluation_panel

    @property
    def uncertainty_panel(self):
        """Compatibility accessor: the uncertainty metrics panel lives on the QA panel."""
        return self.qa_panel.uncertainty_panel

    def on_remote_server_changed(self):
        self.selection_panel.populate_apps()

    def enter(self) -> None:
        """
        Overridden AppTemplateWidget entry point.

        Re-initializes parameter node, GUI state and ensures app selection
        is consistent when the widget is shown.
        """
        if self.selection_panel.ui.appComboBox.count == 0:
            self.selection_panel.populate_apps()

        super().enter()
        self.selection_panel.on_app_selected()
        for panel in self._panels:
            panel.enter()

    def exit(self) -> None:  # noqa: A003
        super().exit()
        for panel in self._panels:
            panel.exit()

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
            and self.inference_panel.required_inputs_ok()
        ):
            self.ui.runInferenceButton.toolTip = _("Start inference")
            self.ui.runInferenceButton.enabled = True
        else:
            self.ui.runInferenceButton.toolTip = _("Select input volume")
            self.ui.runInferenceButton.enabled = False

        inference_stack_volume = self.get_parameter_node("InputVolumeSequence")

        # Configure QA button depending on which tab is active
        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            # Reference-based evaluation: the QA panel checks the active mode
            # (static input/reference pair or dynamic evaluation tab)
            if self.qa_panel.evaluation_inputs_ok():
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

    def dispatch_app_changed(self, app) -> None:
        """
        Reconfigure app-dependent widgets owned by the template when the
        selected app changes, then notify every panel.

        Called by the selection panel at the end of ``on_app_selected``.
        """
        # Enable QA sections based on app capabilities (evaluation/uncertainty support)
        has_inference, has_evaluation, has_uncertainty = app.has_capabilities()

        self.ui.evaluationCollapsible.setEnabled(has_evaluation or has_uncertainty)
        if not has_evaluation and not has_uncertainty:
            self.ui.evaluationCollapsible.collapsed = True

        for panel in self._panels:
            panel.on_app_changed(app)

    def cleanup(self) -> None:
        super().cleanup()
        for panel in self._panels:
            panel.cleanup()
