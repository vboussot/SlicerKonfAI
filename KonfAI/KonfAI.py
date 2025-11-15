from platform import node
import pynvml
import slicer
import psutil
import itertools
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget, ScriptedLoadableModule, ScriptedLoadableModuleLogic
from slicer.util import VTKObservationMixin
from slicer.i18n import tr as _
from slicer.i18n import translate
from qt import (
    QIcon,
    QSize,
    QWidget,QVBoxLayout,QTabWidget,QEventLoop,QTimer, QDesktopServices, QUrl, QListWidgetItem, Qt, QMenu, QCursor, QFont, QColor, QProcess
)
import vtk
import os
import SimpleITK as sitk
from pathlib import Path
import re
import shutil
from konfai.evaluator import Statistics
from konfai.utils.utils import ModelHF, RepositoryHFError, get_available_models_on_hf_repo
import numpy as np
import sitkUtils
from konfai.utils.dataset import image_to_data, get_infos
            

class KonfAI(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KonfAI")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Deep Learning")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Valentin Boussot (University of Rennes, France)",
        ]
        self.parent.helpText = _(
            """SlicerKonfAI enables fast and configurable deep learning inference directly within 3D Slicer,
                using pretrained <b>KonfAI</b> models hosted on Hugging Face.

                You can:
                <ul>
                <li>Load a pretrained model from Hugging Face or a local directory.</li>
                <li>Apply deep learning inference on images.</li>
                <li>Evaluate predictions against a reference image or label map.</li>
                <li>Compute uncertainty maps for model predictions.</li>
                <li>Use GPU acceleration for real-time inference.</li>
                <li>Export predictions and evaluation results.</li>
                </ul>

                This module is designed for research and prototyping using the KonfAI framework.
                For more information, visit the <a href="https://github.com/vboussot/KonfAI">KonfAI repository</a>.
                """

        )
        self.parent.acknowledgementText = _(
            """
This module was originally developed by Valentin Boussot (University of Rennes, France).
It integrates the KonfAI deep learning framework for medical image.

If you use KonfAI in your research, please cite the following work:  
Boussot V., Dillenseger J.-L.:  
<b>KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.</b>  
<a href="https://arxiv.org/abs/2508.09823">https://arxiv.org/abs/2508.09823</a>
"""
        )

def resourcePath(filename):
    """Return the absolute path of the module ``Resources`` directory."""
    scriptedModulesPath = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scriptedModulesPath, "Resources", filename)

class KonfAIMetricsPanel(QWidget):

    def __init__(self):
        super().__init__()
        ui_widget = slicer.util.loadUI(resourcePath("UI/KonfAIMetricsPanel.ui"))
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.ui.imagesListWidget.itemClicked.connect(self.onImageClicked)

    def clearMetrics(self):
        self.ui.metricsListWidget.clear()

    def addMetric(self, key, value):
        text = f"{key:<15} : {value:.4g}"
        item = QListWidgetItem(text)
        font = QFont("Courier New", 10)
        item.setFont(font)
        item.setForeground(QColor("#111"))
        self.ui.metricsListWidget.addItem(item)

    def setMetrics(self, metrics: dict[str, float]):
        self.clearMetrics()
        for key, value in metrics.items():
            self.addMetric(key, value)

    def onImageClicked(self):
        item = self.ui.imagesListWidget.currentItem()
        fullPath = item.data(Qt.UserRole)
        volumeNode = slicer.util.loadVolume(fullPath)
        _, attr = get_infos(fullPath)
        for key, value in attr.items():
            volumeNode.SetAttribute(key.split("_")[0], str(value))
        slicer.util.setSliceViewerLayers(background=volumeNode)

    def clearImagesList(self):
        self.ui.imagesListWidget.clear()

    def refreshImagesList(self, path: Path):
        self.ui.imagesListWidget.clear()
        for filename in sorted(list(path.rglob("*.mha"))):
            item = QListWidgetItem(filename.name)
            item.setFont(QFont("Arial", 10))
            item.setForeground(QColor("#222"))
            item.setData(Qt.UserRole, str(filename))
            self.ui.imagesListWidget.addItem(item)

class Process(QProcess):

    def __init__(self, _update_logs, _update_progress):
        super().__init__(self)
        self.readyReadStandardOutput.connect(self.on_stdout_ready)
        self.readyReadStandardError.connect(self.on_stderr_ready)

        self._update_logs = _update_logs
        self._update_progress = _update_progress

    def on_stdout_ready(self):
        line = self.readAllStandardOutput().data().decode().strip()
        if line:
            line = line.replace('\r\n', '\n').split('\r')[-1]

            self._update_logs(line)
            m = re.search(r"\b(\d{1,3})%(?=\s+\d+/\d+)", line)
            speed = re.search(r"([\d.]+)\s*(it/s|s/it)", line)
            if m:
                pct = int(m.group(1))
            if speed:
                self._update_progress(pct, speed.group(1) + " " + speed.group(2))
    
    def on_stderr_ready(self):
        print("Error : ", self.readAllStandardError().data().decode().strip())

    def run(self, work_dir: Path, args: list[str], on_end_function):
        self.setWorkingDirectory(str(work_dir))
        self.finished.connect(on_end_function)

        self.start("konfai", args)
    
    def stop(self):
        self.kill()
        self.waitForFinished(-1)


class KonfAIAppTemplateWidget(QWidget):

    def __init__(self, name: str, konfai_repo_list: list[str]):
        super().__init__()
        self.name = name
        self.proc = None
        self._update_logs = None
        self._update_progress = None
        self._parameterNode = None
        self._work_dir = None

        ui_widget = slicer.util.loadUI(resourcePath("UI/KonfAIAppTemplate.ui")) 
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        self.evaluationPanel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluationPanel)
        self.uncertaintyPanel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertaintyPanel)

        ui_widget.setMRMLScene(slicer.mrmlScene)

        self.description_expanded = False
        for konfai_repo in konfai_repo_list:
            model_names = []
            try:
                model_names = get_available_models_on_hf_repo(konfai_repo)
            except RepositoryHFError as e:
                slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
            for model_name in get_available_models_on_hf_repo(konfai_repo):
                try:
                    model = ModelHF(konfai_repo + ":" + model_name)
                    self.ui.modelComboBox.addItem(model.get_display_name(), model)
                except RepositoryHFError as e:
                    slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
        self.ui.modelComboBox.setCurrentIndex(0)
        
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)

        self.ui.inputVolumeEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.on_inputVolumeEvaluation_changed)
        self.ui.inputInferenceStackSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.on_inputVolumeEvaluation_changed)

        self.ui.referenceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.referenceMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.transformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.modelComboBox.currentIndexChanged.connect(self.on_model_selected)

        self.ui.addModelButton.clicked.connect(self.on_add_model)
        self.ui.removeModelButton.clicked.connect(self.on_remove_model)

        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "gear.png")
        self.ui.configButton.setIcon(QIcon(iconPath))
        self.ui.configButton.setIconSize(QSize(18, 18))
        self.ui.configButton.clicked.connect(self.on_open_config)

        self.ui.runInferenceButton.clicked.connect(self.on_run_inference_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)
        
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)
    
    def set_information(self, model = None, number_of_ensemble = None, number_of_tta = None, number_of_mc_dropout = None):
        self.ui.modelSummaryValue.setText(f"Model: {model}" if model else "Model: N/A")
        self.ui.ensembleSummaryValue.setText(f"#{number_of_ensemble}" if number_of_ensemble else "#N/A")
        self.ui.ttaSummaryValue.setText(f"#{number_of_tta}" if number_of_ensemble and number_of_tta else "#N/A")
        self.ui.mcSummaryValue.setText(f"#{number_of_mc_dropout}" if number_of_ensemble and number_of_mc_dropout else "#N/A")

    def on_inputVolumeEvaluation_changed(self, node):
        if node:
            storage = node.GetStorageNode()
            if storage:
                path = storage.GetFileName()
                _, attr = get_infos(path)
                if "Model" in attr and "NumberOfEnsemble" in attr and "NumberOfTTA" in attr and "NumberOfMCDropout" in attr:
                    self.set_information(attr["Model"], attr["NumberOfEnsemble"], attr["NumberOfTTA"], attr["NumberOfMCDropout"])
                else:
                    self.set_information()
            else:
                self.set_information(node.GetAttribute("Model"), node.GetAttribute("NumberOfEnsemble"), node.GetAttribute("NumberOfTTA"), node.GetAttribute("NumberOfMCDropout"))

        self.update_parameter_node_from_GUI()

    def konfai_app_setup(self, update_logs, update_progress, parameterNode):
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameterNode = parameterNode
        self.process = Process(update_logs, update_progress)
        
    def enter(self):
        self.initialize_parameter_node()
        self.initialize_GUI_from_parameter_node()
        self.update_GUI_from_parameter_node()
        self.on_model_selected()

    def initialize_parameter_node(self):
        if not self._parameterNode.GetNodeReference(f"{self.name}/InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID(f"{self.name}/InputVolume", firstVolumeNode.GetID())
        if not self._parameterNode.GetParameter(f"{self.name}/Model"):
            model = self.ui.modelComboBox.itemData(0)        
            self._parameterNode.SetParameter(f"{self.name}/Model", f"{model.repo_id}:{model.model_name}")

        current_model = None
        for model in [self.ui.modelComboBox.itemData(i) for i in range(self.ui.modelComboBox.count)]:
            if f"{model.repo_id}:{model.model_name}" == self._parameterNode.GetParameter(f"{self.name}/Model"):
                current_model = model
                break
        if not current_model:
             current_model = self.ui.modelComboBox.itemData(0)

        if not self._parameterNode.GetParameter(f"{self.name}/number_of_ensemble"):
            self._parameterNode.SetParameter(f"{self.name}/number_of_ensemble", str(current_model.get_number_of_models()))
        if not self._parameterNode.GetParameter(f"{self.name}/number_of_tta"):    
            self._parameterNode.SetParameter(f"{self.name}/number_of_tta", str(current_model.get_maximum_tta()))
        if not self._parameterNode.GetParameter(f"{self.name}/number_of_mc_dropout"):
            self._parameterNode.SetParameter(f"{self.name}/number_of_mc_dropout", str(current_model._mc_dropout))

    def initialize_GUI_from_parameter_node(self):
        self.ui.inputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/InputVolume"))
        
        index = self.ui.modelComboBox.findData(self._parameterNode.GetParameter(f"{self.name}/Model"))
        self.ui.modelComboBox.setCurrentIndex(index if index != -1 else 0)

        self.ui.outputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/OutputVolume"))
        self.ui.ensembleSpinBox.setValue(int(self._parameterNode.GetParameter(f"{self.name}/number_of_ensemble")))
        self.ui.ttaSpinBox.setValue(int(self._parameterNode.GetParameter(f"{self.name}/number_of_tta")))
        self.ui.mcDropoutSpinBox.setValue(int(self._parameterNode.GetParameter(f"{self.name}/number_of_mc_dropout")))
        
        self.ui.transformSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/Transform"))

    def update_GUI_from_parameter_node(self):
        inputVolume = self._parameterNode.GetNodeReference(f"{self.name}/InputVolume")
        if inputVolume and self.ui.modelComboBox.currentData:
            self.ui.runInferenceButton.toolTip = _("Start inference")
            self.ui.runInferenceButton.enabled = True
        else:
            self.ui.runInferenceButton.toolTip = _("Select input volume")
            self.ui.runInferenceButton.enabled = False

        if not self.is_running():
            self.ui.runInferenceButton.text = "Run"
            self.ui.runEvaluationButton.text = "Run"            
        else:
            self.ui.runInferenceButton.text = "Stop"
            self.ui.runEvaluationButton.text = "Stop"

        referenceVolume = self._parameterNode.GetNodeReference(f"{self.name}/ReferenceVolume")
        inputEvaluationVolume = self._parameterNode.GetNodeReference(f"{self.name}/InputEvaluationVolume")
        inferenceStackVolume = self._parameterNode.GetNodeReference(f"{self.name}/InputInferenceStackVolume")

        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            if referenceVolume and referenceVolume.GetImageData() and inputEvaluationVolume and inputEvaluationVolume.GetImageData():
                self.ui.runEvaluationButton.toolTip = _("Start evaluation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input and reference volumes")
                self.ui.runEvaluationButton.enabled = False
        else:
            if inferenceStackVolume and inferenceStackVolume.GetImageData() and inferenceStackVolume.GetImageData().GetNumberOfScalarComponents() > 1:
                self.ui.runEvaluationButton.toolTip = _("Start uncertainty estimation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input volume")
                self.ui.runEvaluationButton.enabled = False

        if inputVolume:
            self.ui.outputVolumeSelector.baseName = _("{volume_name} Output").format(volume_name=inputVolume.GetName())

    def update_parameter_node_from_GUI(self, caller=None, event=None):
        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
        self._parameterNode.SetNodeReferenceID(f"{self.name}/InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetParameter(f"{self.name}/Model", str(self.ui.modelComboBox.currentIndex))
        self._parameterNode.SetNodeReferenceID(f"{self.name}/OutputVolume", self.ui.outputVolumeSelector.currentNodeID)
        
        self._parameterNode.SetNodeReferenceID(f"{self.name}/ReferenceVolume", self.ui.referenceVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/ReferenceMask", self.ui.referenceMaskSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/Transform", self.ui.transformSelector.currentNodeID)
        
        self._parameterNode.SetNodeReferenceID(f"{self.name}/InputEvaluationVolume", self.ui.inputVolumeEvaluationSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/InputInferenceStackVolume", self.ui.inputInferenceStackSelector.currentNodeID)
        self._parameterNode.EndModify(wasModified)

    def _save_models_dir_list_in_settings(self) -> None:
        settings = QSettings()
        # settings.setValue(SETTING_MODEL_DIR_KEY, json.dumps(list(self.models.keys())))

    def on_model_selected(self):
        """Handle model selection and display its description."""
        model: ModelHF = self.ui.modelComboBox.currentData
        self.ui.removeModelButton.setEnabled(False)
        self.description_expanded = False
        self.on_toggle_description()

        self.ui.ensembleSpinBox.setMaximum(model.get_number_of_models())
        self.ui.ensembleSpinBox.setValue(model.get_number_of_models())
        self.ui.ttaSpinBox.setEnabled(model.get_maximum_tta() > 0)
        self.ui.ttaSpinBox.setMaximum(model.get_maximum_tta())

        self.ui.mcDropoutSpinBox.setEnabled(model._mc_dropout > 0)
        self.ui.mcDropoutSpinBox.setMaximum(model._mc_dropout)
        has_evaluation, has_uncertainty = model.has_capabilities()
        self.ui.evaluationCollapsible.setEnabled(has_evaluation or has_uncertainty)
        
        self.ui.qaTabWidget.setTabEnabled(0, has_evaluation)
        self.ui.qaTabWidget.setTabEnabled(1, has_uncertainty)

    def on_toggle_description(self):
        model = self.ui.modelComboBox.currentData
        if self.description_expanded:
            self.ui.modelDescriptionLabel.setText(model.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.modelDescriptionLabel.setText(model.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")
        self.description_expanded = not self.description_expanded

    def on_remove_model(self):
        cb = self.ui.modelComboBox
        idx = cb.currentIndex
        model_dir = self.ui.modelComboBox.currentData

        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("Remove model?")
        mb.setText(f"Do you really want to remove “{model_dir}” from the list?")
        mb.setInformativeText("This will remove the model entry from the extension’s list.")
        mb.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        mb.setDefaultButton(QMessageBox.Cancel)

        chk = QCheckBox("Also delete the folder from disk")
        chk.setChecked(False)
        mb.setCheckBox(chk)

        if mb.exec_() != QMessageBox.Yes:
            return

        cb.removeItem(idx)

        if chk.isChecked() and model_dir and os.path.isdir(model_dir):
            try:
                dangerous = {os.path.expanduser("~"), "/", "C:\\"}
                if os.path.abspath(model_dir) in dangerous:
                    raise RuntimeError(f"Refusing to delete a critical directory: {model_dir}")

                shutil.rmtree(model_dir)
                QMessageBox.information(
                    None, "Folder deleted", f"The folder has been successfully deleted:\n{model_dir}"
                )
            except Exception as e:
                QMessageBox.critical(None, "Deletion error", f"Failed to delete folder:\n{model_dir}\n\n{e}")

    def on_open_config(self):
        """
        Open configuration file when user clicks "Open config" button.
        """
        _, inference_file_path, _ = self.ui.modelComboBox.currentData.download(0)
        QDesktopServices.openUrl(QUrl.fromLocalFile(Path(inference_file_path).parent))

    def on_add_model(self):
        m = QMenu()
        act_folder = m.addAction("Add from folder…")
        act_hf = m.addAction("Add from Hugging Face…")
        act_ft = m.addAction("Setup fine-tuning")
        chosen = m.exec_(QCursor.pos())
        if chosen is None:
            return
        """if chosen is act_folder:
            self.on_add_folder()
        elif chosen is act_hf:
            self.on_add_hf()
        elif chosen is act_ft:
            self.on_add_ft()"""

    def on_add_folder(self):
        model_dir = QFileDialog.getExistingDirectory(None, "Select Model Folder", os.path.expanduser("~"))
        if not model_dir:
            return
        if model_dir + ":None" in self.models:
            QMessageBox.information(None, "Info", f"This name {model_dir} is already in the list.")
        try:
            model = Model(Path(model_dir), None)
        except ModelConfigError as e:
            slicer.util.errorDisplay(str(e), detailedText=getattr(e, "details", None) or "")
            return

        for model_tmp in self.models.values():
            if model_tmp.get_display_name() == model.get_display_name():
                QMessageBox.warning(
                    None,
                    "Duplicate Model Name",
                    f"A model named “{model_tmp.get_display_name()}” already exists in the list.\n\n"
                    "Each model must have a unique display name.",
                )
                return
        self.models[model_dir + ":None"] = model
        self.ui.modelComboBox.addItem(model.get_display_name(), model_dir + ":None")
        self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.findData(model_dir + ":None"))

    def on_add_hf(self):
        text = QInputDialog.getText(
            None, "Add from Hugging Face", "Enter repo id (e.g. org/model or org/model@rev):", QLineEdit.Normal
        )
        if not text.strip():
            return
        repo_id = text.strip()
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            try:
                api.model_info(repo_id.split("@")[0])  
            except Exception:
                api.dataset_info(repo_id.split("@")[0]) 
        except Exception:
            pass

    def on_add_ft(self):
        # Choisir le dossier parent
        parent_dir = QFileDialog.getExistingDirectory(self.parent, "Choose parent directory for the new model")
        if not parent_dir:
            return

        # Nom du nouveau modèle
        name, ok = QInputDialog.getText(self.parent, "Fine tune", "New model name (folder will be created):")
        if not ok or not name.strip():
            return
        # Slug safe
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        if not safe:
            QMessageBox.warning(self.parent, "Invalid name", "Please enter a valid name.")
            return

        # Créer le dossier (ajoute suffixe _1, _2 si existe)
        base = Path(parent_dir)
        target = base / safe
        if target.exists():
            i = 1
            while (base / f"{safe}_{i}").exists():
                i += 1
            target = base / f"{safe}_{i}"
        try:
            target.mkdir(parents=True, exist_ok=False)
            (target / "README.txt").write_text("Fine-tune model folder (empty).\n")
        except Exception as e:
            QMessageBox.critical(self.parent, "Create folder failed", str(e))
            return

        # Ajoute dans la combo
        self.ui.modelComboBox.addItem(f"{target.name} (ft)", str(target))
        self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.count() - 1)
        if hasattr(self.logic, "logCallback") and self.logic.logCallback:
            self.logic.logCallback(f"Fine-tune folder created: {target}")

    def has_uncertainty_file(self):
        return self._work_dir and any((self._work_dir / "Uncertainty").rglob("*.mha"))

    def get_work_dir(self) -> Path | None:
        return self._work_dir
    
    def remove_work_dir(self):
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir)
            self._work_dir = None
    
    def is_running(self) -> bool:
        return self._parameterNode.GetParameter("is_running") == "True"
    
    def set_running(self, state: bool) -> None:
        self._parameterNode.SetParameter("is_running", str(state))


    def on_tab_changed(self):
        self.update_GUI_from_parameter_node()

    def on_run_button(self, function):
        """
        Run processing when user clicks "Apply" button.
        """
        if not self.is_running():
            self.set_running(True)
            self._update_logs("Processing started.", True)
            self._update_progress(0, "0 it/s")
            try:
                function()
            except Exception as e:
                print(e)
                self.set_running(False)
        else:
            self.ui.runInferenceButton.setEnabled(False)
            self.ui.runEvaluationButton.setEnabled(False) 

            self.set_running(False)
            self.process.stop()
            import time
            time.sleep(3)
            self.ui.runInferenceButton.setEnabled(True)
            self.ui.runEvaluationButton.setEnabled(True) 

            
    def on_run_inference_button(self):
        self.on_run_button(self.inference)
    
    def on_run_evaluation_button(self):
        self.on_run_button(self.evaluation if self.ui.qaTabWidget.currentWidget().name == "withRefTab" else self.uncertainty)


    def evaluation(self):
        self.evaluationPanel.clearImagesList()
        self.evaluationPanel.clearMetrics()
        if not self.ui.transformSelector.currentNode():
            newTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "IdentityTransform")
            self.ui.transformSelector.setCurrentNode(newTransform)
            self._parameterNode.SetNodeReferenceID(f"{self.name}/Transform", self.ui.transformSelector.currentNodeID)

        outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.ui.inputVolumeEvaluationSelector.currentNode().GetName() + "_toRef")

        params = {
            "inputVolume": self.ui.inputVolumeEvaluationSelector.currentNode().GetID(),
            "referenceVolume": self.ui.referenceVolumeSelector.currentNode().GetID(),
            "outputVolume": outputNode.GetID(),
            "interpolationType": "linear", 
            "warpTransform": self.ui.transformSelector.currentNode().GetID(),
        }

        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)


        dataset_p = self._work_dir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(str(dataset_p / "Output.mha"))
        volumeStorageNode.UseCompressionOff()
        volumeStorageNode.WriteData(outputNode)
        volumeStorageNode.UnRegister(None)

        dataset_p = self._work_dir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(str(dataset_p / "Reference.mha"))
        volumeStorageNode.UseCompressionOff()
        volumeStorageNode.WriteData(self.ui.referenceVolumeSelector.currentNode())
        volumeStorageNode.UnRegister(None)

        if not self.ui.referenceMaskSelector.currentNode() or not self.ui.referenceMaskSelector.currentNode().GetImageData():
            refVolume = self.ui.referenceVolumeSelector.currentNode()
            refImage = refVolume.GetImageData()
            maskImage = vtk.vtkImageData()
            maskImage.DeepCopy(refImage)
            maskImage.GetPointData().GetScalars().Fill(1)

            maskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Mask")
            maskVolume.SetAndObserveImageData(maskImage)
            mat = vtk.vtkMatrix4x4()
            refVolume.GetIJKToRASMatrix(mat)
            maskVolume.SetIJKToRASMatrix(mat)
            maskVolume.SetSpacing(refVolume.GetSpacing())
        else:
            maskVolume = self.ui.referenceMaskSelector.currentNode()

        maskStorage = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        maskStorage.SetFileName(str(dataset_p / "Mask.mha"))
        maskStorage.UseCompressionOff()
        maskStorage.WriteData(maskVolume)
        maskStorage.UnRegister(None)

        args = [
            "EVALUATION",
            "-y",
        ]
        if self._parameterNode.GetParameter("Device") != "None":
            args += ["--gpu", self._parameterNode.GetParameter("Device")]
        else:
            args += ["--cpu", "1"]
        
        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                return
            statistics = Statistics(str(self._work_dir / "Evaluations" / "Evaluation" / "Metric_TRAIN.json"))
            self.evaluationPanel.setMetrics(statistics.read()) 
            self.evaluationPanel.refreshImagesList(self._work_dir / "Evaluation")
            self._update_logs("Processing finished.")
            self.set_running(False)
            
        self.process.run(self._work_dir, args, on_end_function)

        
    def uncertainty(self):
        self.uncertaintyPanel.refreshImagesList(self._work_dir / "Uncertainty")
        self.uncertaintyPanel.clearMetrics()
        args = [
            "EVALUATION",
            "-y",
            "--config",
            "Uncertainty.yml",
        ]
        if self._parameterNode.GetParameter("Device") != "None":
            args += ["--gpu", self._parameterNode.GetParameter("Device")]
        else:
            args += ["--cpu", "1"]

        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                return
            statistics = Statistics(str(self._work_dir / "Evaluations" / "Uncertainty" / "Metric_TRAIN.json"))
            self.uncertaintyPanel.setMetrics(statistics.read()) 
            self.uncertaintyPanel.refreshImagesList(self._work_dir / "Uncertainty")
            self._update_logs("Processing finished.")
            self.set_running(False)

        self.process.run(self._work_dir, args, on_end_function)

    def inference(self) -> None:
        self.remove_work_dir()
        self._work_dir = Path(slicer.util.tempDirectory())

        self.set_running(True)
        model: ModelHF = self.ui.modelComboBox.currentData
        args = [
            "PREDICTION_HF",
            "-y",
            "--MODEL",
            str(self.ui.ensembleSpinBox.value),
            "--tta",
            str(self.ui.ttaSpinBox.value),
            "--config",
            f"{model.repo_id}:{model.model_name}",
        ]
        if self._parameterNode.GetParameter("Device") != "None":
            args += ["--gpu", self._parameterNode.GetParameter("Device")]
        else:
            args += ["--cpu", "1"]

        dataset_p = self._work_dir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)


        inputNode = self.ui.inputVolumeSelector.currentNode()

        sitk_image = sitkUtils.PullVolumeFromSlicer(inputNode)
        sitk_image.SetMetaData("Model", f"{self.ui.modelComboBox.currentData.repo_id}:{self.ui.modelComboBox.currentData.model_name}")
        sitk_image.SetMetaData("NumberOfEnsemble", f"{self.ui.ensembleSpinBox.value}")
        sitk_image.SetMetaData("NumberOfTTA", f"{self.ui.ttaSpinBox.value}")
        sitk_image.SetMetaData("NumberOfMCDropout", f"{self.ui.mcDropoutSpinBox.value}")

        path = str(dataset_p / "Volume.mha")
        sitk.WriteImage(sitk_image, path)

        self._update_logs(f"Input volume saved to temporary folder: {dataset_p}")

        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                return

            data, attr = image_to_data(sitk.ReadImage(str(list((self._work_dir / "Predictions").rglob("*.mha"))[0])))            

            self._update_logs("Loading result into Slicer...")

            want_label = data.dtype == np.uint8
            expected_class = "vtkMRMLLabelMapVolumeNode" if want_label else "vtkMRMLScalarVolumeNode"
            base_name = "OutputLabel" if want_label else "OutputVolume"

            current = self.ui.outputVolumeSelector.currentNode()

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

            slicer.util.updateVolumeFromArray(node, data[0])
            node.CopyOrientation(self.ui.inputVolumeSelector.currentNode())
            for key, value in attr.items():
                node.SetAttribute(key.split("_")[0], str(value))

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
            self.evaluationPanel.refreshImagesList(self._work_dir / "Evaluation") 
            self.uncertaintyPanel.refreshImagesList(self._work_dir / "Uncertainty")

            self._update_logs("Processing finished.")
            self.set_running(False)

        self.process.run(self._work_dir, args, on_end_function)

class KonfAICoreWidget(QWidget, VTKObservationMixin, ScriptedLoadableModuleLogic):

    def __init__(self, title: str) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        QWidget.__init__(self)
        VTKObservationMixin.__init__(self)
        ScriptedLoadableModuleLogic.__init__(self)
        
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._konfai_apps = {}
        self._current_konfai_app = None
        
        ui_widget = slicer.util.loadUI(resourcePath("UI/KonfAICore.ui")) 
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.initializeParameterNode()

        self.ui.headerTitleLabel.text = title

        available_devices = self._get_available_devices()
        for available_device in available_devices:
            self.ui.deviceComboBox.addItem(available_device[0], available_device[1])

        self.ui.deviceComboBox.currentIndexChanged.connect(self.on_device_changed)
        self.ui.deviceComboBox.setCurrentIndex(0 if len(available_devices) == 0 else 1)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.on_scene_start_close)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.on_scene_end_close)

        self.ui.openTempButton.setIcon(QIcon(resourcePath("Icons/folder.png")))
        self.ui.openTempButton.setIconSize(QSize(18, 18))
        self.ui.openTempButton.clicked.connect(self.on_open_work_dir)
        self.ui.openTempButton.setEnabled(False)
        
        self.update_logs("", True)        

    def register_konfai_apps(self, konfai_apps: list[QWidget]):
        if len(konfai_apps) > 1:
            tabWidget = QTabWidget()
            for konfai_app in konfai_apps:
                tabWidget.addTab(konfai_app, konfai_app.name)
                
            def on_tab_changed(self):
                tabWidget.currentWidget().enter()
            
            tabWidget.currentChanged.connect(on_tab_changed)
            
            self.KonfAICoreWidget.layout().insertWidget(1, tabWidget)
        else:
            konfai_app = konfai_apps[0]
            self.KonfAICoreWidget.layout().insertWidget(1, konfai_app)
                    
        for konfai_app in konfai_apps:
            self._konfai_apps[konfai_app.name] = konfai_app
            konfai_app.konfai_app_setup(self.update_logs, self.update_progress, self._parameterNode)

        konfai_app = next(iter(self._konfai_apps.values()))
        self._current_konfai_app = konfai_app
        konfai_app.enter()


    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        self.setParameterNode(self.getParameterNode())
        self._parameterNode.SetParameter("is_running", "False")
        self.update_GUI_from_parameter_node()
    
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.

        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.update_GUI_from_parameter_node)

        self._parameterNode = inputParameterNode

        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.update_GUI_from_parameter_node)
    
    def update_GUI_from_parameter_node(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode and not self._updatingGUIFromParameterNode and self._current_konfai_app:
            self._updatingGUIFromParameterNode = True
            self._current_konfai_app.update_GUI_from_parameter_node()
            self._updatingGUIFromParameterNode = False

            self._parameterNode.GetParameter("is_running")
            
            self.ui.openTempButton.setEnabled(self._current_konfai_app.get_work_dir() is not None)

    def on_scene_start_close(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def on_scene_end_close(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode() 
    
    def _get_available_devices(self) -> list[tuple[str, str | None]]:
        available_devices = [("cpu [slow]", None)]
        try:
            from torch.cuda import device_count, get_device_name, is_available
        except:
            slicer.util.pip_install("konfai")

        if is_available():
            combos = []
            nb_gpu = device_count()
            for r in range(1, nb_gpu + 1):
                combos.extend(itertools.combinations(range(nb_gpu), r))
            for device in combos:
                device_name = get_device_name(device[0])
                index = str(device[0])
                for i in device[1:]:
                    deviceName += f",{get_device_name(i)}"
                    index += f"-{i}"
                available_devices.append((f"gpu {index} - {device_name}", index))
        return available_devices
    
    def on_device_changed(self):
        self._update_VRAM()
        self._parameterNode.SetParameter("Device", self.ui.deviceComboBox.currentData)

    def on_open_work_dir(self):
        """
        Open inference work_dir when user clicks "Open workdir" button.
        """
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_konfai_app.get_work_dir()))
    
    def _update_ram(self):
        """Update RAM usage display"""
        ram = psutil.virtual_memory()
        used_GB = (ram.total - ram.available) / (1024**3)
        total_GB = ram.total / (1024**3)
        self.ui.ramLabel.text = _("RAM used: {used:.1f} GB / {total:.1f} GB").format(used=used_GB, total=total_GB)
        self.ui.ramProgressBar.value = used_GB / total_GB * 100
    
    def _update_VRAM(self):
        """Update VRAM usage display"""
        device = self.ui.deviceComboBox.currentData
        if device is not None:
            try:
                used_GB = 0
                total_GB = 0
                pynvml.nvmlInit()
                for index in device.split(","):
                    info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(int(index)))
                    used_GB += info.used / (1024**3)
                    total_GB += info.total / (1024**3)
                self.ui.gpuLabel.show()
                self.ui.gpuProgressBar.show()
                self.ui.gpuLabel.text = _("VRAM used: {used:.1f} GB / {total:.1f} GB").format(
                    used=used_GB, total=total_GB
                )
                self.ui.gpuProgressBar.value = used_GB / total_GB * 100
            except Exception as e:
                self.ui.gpuLabel.text = _("VRAM used: n/a")
        else:
            self.ui.gpuLabel.hide()
            self.ui.gpuProgressBar.hide()
    
    def update_logs(self, text: str, clear: bool = False) -> None:
        """Append text to log window"""
        self._update_ram()
        self._update_VRAM()
        if clear:
            self.ui.logText.plainText = text
        else:
            self.ui.logText.appendPlainText(text)

    def update_progress(self, value: int, speed: float) -> None:
        """Update progress bar"""
        self._update_ram()
        self._update_VRAM()
        self.ui.progressBar.value = value
        self.ui.speedLabel.text = _("{speed}").format(speed=speed)
    
class KonfAIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
    
    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)
        self.konfai_core = KonfAICoreWidget("KonfAI")
        predictionWidget = KonfAIAppTemplateWidget("Inference", ["VBoussot/ImpactSynth", "VBoussot/MRSegmentator-KonfAI"])
        self.konfai_core.register_konfai_apps([predictionWidget])
        self.layout.addWidget(self.konfai_core)
        
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        pass

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        pass