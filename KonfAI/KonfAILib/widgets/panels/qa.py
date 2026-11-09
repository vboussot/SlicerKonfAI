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

"""QA panel: reference-based evaluation and reference-free uncertainty
estimation, with their metrics panels and the app information summary."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import slicer
import vtk
from qt import QFormLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from KonfAILib.logic.mrml import has_node_content
from KonfAILib.logic.servers import RemoteServer
from KonfAILib.widgets import io_selectors
from KonfAILib.widgets.metrics_panel import KonfAIMetricsPanel
from KonfAILib.widgets.panels.base import KonfAIAppPanel

if TYPE_CHECKING:
    from KonfAILib.widgets.app_template import KonfAIAppTemplateWidget


def node_image(node) -> sitk.Image:
    """``node`` as a SimpleITK image in LPS, a vector volume too: sitkUtils.PullVolumeFromSlicer reads a scalar
    volume only. The voxel values are the node's own, components included."""
    array = slicer.util.arrayFromVolume(node)
    image = sitk.GetImageFromArray(array, isVector=array.ndim == 4)
    ijk_to_ras = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijk_to_ras)
    ijk_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ slicer.util.arrayFromVTKMatrix(ijk_to_ras)
    spacing = np.array(node.GetSpacing())
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(ijk_to_lps[:3, 3].tolist())
    image.SetDirection((ijk_to_lps[:3, :3] / spacing).ravel().tolist())
    return image


class KonfAIAppQAPanel(KonfAIAppPanel):
    """
    QA panel of a KonfAI application.

    This panel owns the QA tab widget (reference-based evaluation and
    reference-free uncertainty estimation), the evaluation/uncertainty
    metrics panels, the app information summary row and the QA run button.
    """

    def __init__(self, template: "KonfAIAppTemplateWidget") -> None:
        super().__init__(template, "KonfAIAppQA")

        # Attach metrics panels in both QA contexts: reference-based and reference-free
        self.evaluation_panel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluation_panel)
        self.uncertainty_panel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertainty_panel)

        # Document-mode tabs: flat tab bars without the heavy frame around the page, so the two
        # nested tab levels stop reading as boxes-in-boxes.
        self.ui.qaTabWidget.setDocumentMode(True)

        # The run-context summary (App / Ensemble / TTA / MC) reads as subtle pills, not raised boxes.
        for frame in (
            self.ui.appSummaryFrame,
            self.ui.ensembleSummaryFrame,
            self.ui.ttaSummaryFrame,
            self.ui.mcSummaryFrame,
        ):
            frame.setStyleSheet(
                "QFrame { background: rgba(128,128,128,0.12); border: none; border-radius: 10px; }"
            )

        # Internal tabs: one per EvaluationKey declared by the app manifest;
        # hidden in static (single evaluation) mode.
        self._evaluation_tabs = QTabWidget()
        self._evaluation_tabs.setDocumentMode(True)
        self._evaluation_tabs.setVisible(False)
        self.ui.evaluationTabsPlaceholder.layout().addWidget(self._evaluation_tabs)
        self._evaluation_tabs.currentChanged.connect(self.on_tab_changed)
        # {EvaluationKey: {key: (required, qMRMLNodeComboBox)}} in manifest order
        self._evaluation_input_selectors: dict = {}

        # The same for the uncertainty configs (inputs_uncertainties): one tab per config, the references it reads
        # beside the stack.
        self._uncertainty_tabs = QTabWidget()
        self._uncertainty_tabs.setDocumentMode(True)
        self._uncertainty_tabs.setVisible(False)
        self.ui.formLayout_qa_noref.insertRow(1, self._uncertainty_tabs)
        self._uncertainty_tabs.currentChanged.connect(self.on_tab_changed)
        self._uncertainty_input_selectors: dict = {}

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
        self.ui.referenceMaskSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.template.update_parameter_node_from_gui
        )
        self.ui.inputTransformSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.template.update_parameter_node_from_gui
        )

        # Run button for QA
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        # QA tab changes
        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)

    def on_app_changed(self, app) -> None:
        """
        Reconfigure the QA tabs when the selected app changes.

        Called by the template's ``dispatch_app_changed``.
        """
        has_inference, has_evaluation, has_uncertainty = app.has_capabilities()

        # Enable/disable tabs
        self.ui.qaTabWidget.setTabEnabled(0, has_evaluation)
        self.ui.qaTabWidget.setTabEnabled(1, has_uncertainty)

        self._rebuild_evaluation_tabs(app)
        tabs = app.get_uncertainties_inputs()
        self._uncertainty_input_selectors = self._fill_tabs(self._uncertainty_tabs, tabs, "Unc")
        self._uncertainty_tabs.setVisible(bool(tabs))
        # The selectors' signals fired while the previous app's were still in place: check Run on the new ones.
        self.template.update_gui_from_parameter_node()

    def _fill_tabs(self, tab_widget, tabs: dict, prefix: str) -> dict:
        """One tab of ``tab_widget`` per config of ``tabs`` ({EvaluationKey: entries}), a node selector per entry,
        each selection kept under ``{prefix}/{config file}/{entry}``. Returns {EvaluationKey: {entry: (required,
        selector)}} in the manifest order."""
        # QTabWidget.clear() does not destroy the pages: delete them explicitly
        # so their qMRMLNodeComboBox children stop observing the scene.
        for i in range(tab_widget.count):
            tab_widget.widget(i).deleteLater()
        tab_widget.clear()
        selectors_by_key = {}
        for key, entries in tabs.items():
            tab = QWidget()
            root = QVBoxLayout(tab)
            root.setContentsMargins(12, 12, 12, 12)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            selectors = io_selectors.add_inputs_on_layout(form, entries)
            if not entries:
                form.addRow(QLabel("No reference: the run's output alone."))
            root.addLayout(form)
            root.addStretch(1)
            tab_widget.addTab(tab, key.display_name)
            selectors_by_key[key] = selectors
            for name, (_required, selector) in selectors.items():
                param_key = f"{prefix}/{key.evaluation_file}/{name}"
                selector.connect(
                    "currentNodeChanged(vtkMRMLNode*)",
                    lambda node, pk=param_key: self._on_evaluation_selector_changed(pk, node),
                )
                selector.setCurrentNode(self.get_parameter_node(param_key))
        return selectors_by_key

    def _rebuild_evaluation_tabs(self, app) -> None:
        """
        Regenerate one internal tab of dynamic selectors per declared evaluation.

        Apps without declared evaluations keep the static selector rows
        (current behavior); otherwise the static input/reference rows are hidden.
        """
        evaluations_inputs = dict(getattr(app, "get_evaluations_inputs", dict)())

        # The positional CLI routing needs an input entry per evaluation (a measure of the
        # prediction alone, a field's Jacobian, has no reference); otherwise keep the static fallback.
        dynamic = bool(evaluations_inputs) and all(len(inputs) >= 1 for inputs in evaluations_inputs.values())
        self._evaluation_input_selectors = self._fill_tabs(
            self._evaluation_tabs, evaluations_inputs if dynamic else {}, "Eval"
        )
        self._evaluation_tabs.setVisible(dynamic)
        for widget in (
            self.ui.label_outputVolume,
            self.ui.inputVolumeEvaluationSelector,
            self.ui.label_refVolume,
            self.ui.referenceVolumeSelector,
        ):
            widget.setVisible(not dynamic)
        if not dynamic:
            self.ui.label_refMask.setVisible(True)
            self.ui.referenceMaskSelector.setVisible(True)
            return

        # The shared static mask row only stays when no evaluation declares its own Mask_* entry
        declares_mask = any(
            key.startswith("Mask") for selectors in self._evaluation_input_selectors.values() for key in selectors
        )
        self.ui.label_refMask.setVisible(not declares_mask)
        self.ui.referenceMaskSelector.setVisible(not declares_mask)

    def _on_evaluation_selector_changed(self, param_key: str, node) -> None:
        """Persist a dynamic evaluation selector change under its namespaced key."""
        if self.template._parameter_node is None or not self.template._initialized:
            return
        self.set_parameter_node(param_key, node.GetID() if node is not None else None)

    def _current_evaluation_key(self):
        """Return the EvaluationKey of the current internal tab, or None in static mode."""
        keys = list(self._evaluation_input_selectors.keys())
        index = self._evaluation_tabs.currentIndex
        if 0 <= index < len(keys):
            return keys[index]
        return None

    def _current_uncertainty_key(self):
        """The EvaluationKey of the current uncertainty tab, or None for an app that declares none."""
        keys = list(self._uncertainty_input_selectors.keys())
        index = self._uncertainty_tabs.currentIndex
        return keys[index] if 0 <= index < len(keys) else None

    def uncertainty_inputs_ok(self) -> bool:
        """Whether the current uncertainty tab has every reference it requires."""
        key = self._current_uncertainty_key()
        return key is None or all(
            not required or io_selectors.has_selection(selector)
            for required, selector in self._uncertainty_input_selectors[key].values()
        )

    def evaluation_inputs_ok(self) -> bool:
        """
        Return True when the active evaluation has all its required inputs.

        Static mode keeps the historical rule (input and reference volumes);
        dynamic mode checks the required selectors of the current internal tab.
        """
        evaluation_key = self._current_evaluation_key()
        if evaluation_key is None:
            return has_node_content(self.get_parameter_node("ReferenceVolume")) and has_node_content(
                self.get_parameter_node("InputVolumeEvaluation")
            )
        return all(
            not required or io_selectors.has_selection(selector)
            for required, selector in self._evaluation_input_selectors[evaluation_key].values()
        )

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
        self.template.update_parameter_node_from_gui()

    def on_tab_changed(self) -> None:
        """
        Update GUI state when the user switches between QA tabs.

        Ensures that button enabling/disabling is consistent with the current tab.
        """
        self.template.update_gui_from_parameter_node()

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

    def _write_volume(self, node, path: Path) -> None:
        """Write a volume node to disk without compression (moved from ``evaluation``)."""
        volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volume_storage_node.SetFileName(str(path))
        volume_storage_node.UseCompressionOff()
        volume_storage_node.WriteData(node)
        volume_storage_node.UnRegister(None)

    def _save_entry(self, key: str, node) -> str:
        """Write ``node`` as it is under the work folder, named after its entry: a transform as ``.h5``, landmarks
        as ``.fcsv``, a volume or label map as ``.mha``. Returns the file name."""
        if node.IsA("vtkMRMLTransformNode") or node.IsA("vtkMRMLMarkupsNode"):
            file_name = key + (".h5" if node.IsA("vtkMRMLTransformNode") else ".fcsv")
            slicer.util.saveNode(node, str(self._work_dir / file_name))
        else:
            file_name = f"{key}.mha"
            self._write_volume(node, self._work_dir / file_name)
        return file_name

    def write_stack(self, sequence_node, path: Path) -> None:
        """Write the sequence's members as the stack konfai-apps reads: their channels member-major (a vector
        volume, a displacement field, is one member of several), their count and the field mark in the header."""
        images = [node_image(sequence_node.GetNthDataNode(i)) for i in range(sequence_node.GetNumberOfDataNodes())]
        arrays = []
        for image in images:
            array = sitk.GetArrayFromImage(image)
            arrays.append(array if image.GetNumberOfComponentsPerPixel() > 1 else array[..., None])
        stack = sitk.GetImageFromArray(np.concatenate(arrays, axis=-1), isVector=True)
        stack.CopyInformation(images[0])
        from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE
        from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE

        stack.SetMetaData(MEMBERS_ATTRIBUTE, str(len(images)))
        if sequence_node.GetAttribute("KonfAI.DisplacementField"):
            stack.SetMetaData(DISPLACEMENT_FIELD_ATTRIBUTE, "true")
        sitk.WriteImage(stack, str(path))

    def _export_resampled(self, node, reference_node, transform_node, path: Path, nearest: bool) -> None:
        """Resample ``node`` onto ``reference_node`` through ``transform_node`` and write it to ``path``."""
        output_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode",
            node.GetName() + "_toRef",
        )
        params = {
            "inputVolume": node.GetID(),
            "referenceVolume": reference_node.GetID(),
            "outputVolume": output_node.GetID(),
            "interpolationType": "nn" if nearest else "linear",
            "warpTransform": transform_node.GetID(),
        }
        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)
        self._write_volume(output_node, path)

    def evaluation(self, remote_server: RemoteServer, devices: list[str]) -> None:
        """
        Run reference-based evaluation using the selected app.

        Steps:
          - Resample input volume(s) and optional mask to the reference volume space
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
        transform_node = self.ui.inputTransformSelector.currentNode()

        app = self.template.ui.appComboBox.currentData
        args = ["eval", app.get_name()]

        evaluation_key = self._current_evaluation_key()
        if evaluation_key is None:
            # Static fallback: single input/reference pair (and optional mask)
            input_volume_evaluation_node = self.ui.inputVolumeEvaluationSelector.currentNode()
            reference_volume_evaluation_node = self.ui.referenceVolumeSelector.currentNode()
            self._export_resampled(
                input_volume_evaluation_node,
                reference_volume_evaluation_node,
                transform_node,
                self._work_dir / "Volume.mha",
                nearest=False,
            )
            self._write_volume(reference_volume_evaluation_node, self._work_dir / "Reference.mha")
            args += ["-i", "Volume.mha", "--gt", "Reference.mha", "-o", "Evaluation"]

            # Optional: resample and include the reference mask if defined
            if has_node_content(self.ui.referenceMaskSelector.currentNode()):
                self._export_resampled(
                    self.ui.referenceMaskSelector.currentNode(),
                    reference_volume_evaluation_node,
                    transform_node,
                    self._work_dir / "Mask.mha",
                    nearest=True,
                )
                args += ["--mask", "Mask.mha"]
        else:
            # Dynamic mode: positional routing in the manifest order of the
            # current internal tab (1st entry -> -i, Mask* -> --mask, every
            # other entry -> a --gt group of its own).
            items = list(self._evaluation_input_selectors[evaluation_key].items())
            selectors = dict(items)
            # An evaluation given the transform itself (a registration's) pulls the data through it in its
            # config: every node goes as it is, resampled here it would be moved twice. Otherwise the
            # resampling reference is the node routed to the first --gt (2nd entry).
            takes_transform = any(
                selector.currentNode() is not None and selector.currentNode().IsA("vtkMRMLTransformNode")
                for _required, selector in selectors.values()
            )
            reference_node = items[1][1][1].currentNode() if len(items) > 1 and not takes_transform else None
            if reference_node is not None and not reference_node.IsA("vtkMRMLVolumeNode"):
                reference_node = None
            args += ["-o", "Evaluation"]
            for position, (key, (_required, selector)) in enumerate(items):
                node = selector.currentNode()
                if node is None:
                    continue
                if position == 0:
                    flag = "-i"
                else:
                    flag = "--mask" if key.startswith("Mask") else "--gt"
                plain = node.IsA("vtkMRMLTransformNode") or node.IsA("vtkMRMLMarkupsNode")
                if plain or flag == "--gt" or reference_node is None:
                    file_name = self._save_entry(key, node)
                else:
                    file_name = f"{key}.mha"
                    self._export_resampled(
                        node,
                        reference_node,
                        transform_node,
                        self._work_dir / file_name,
                        nearest=flag == "--mask" or node.IsA("vtkMRMLLabelMapVolumeNode"),
                    )
                args += [flag, file_name]

            # Shared static mask, kept when the evaluation declares no Mask_* entry
            if not any(key.startswith("Mask") for key in selectors) and has_node_content(
                self.ui.referenceMaskSelector.currentNode()
            ):
                mask_node = self.ui.referenceMaskSelector.currentNode()
                if reference_node is not None:
                    self._export_resampled(
                        mask_node, reference_node, transform_node, self._work_dir / "Mask.mha", nearest=True
                    )
                else:
                    self._write_volume(mask_node, self._work_dir / "Mask.mha")
                args += ["--mask", "Mask.mha"]

            args += ["--evaluation_file", evaluation_key.evaluation_file]

        # Select device backend from parameter node (GPU indices or CPU fallback)
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += remote_server.cli_args()

        def on_end_function() -> None:
            """
            Callback executed when the evaluation process finishes.

            It loads metrics and images produced by konfai-apps and updates
            the evaluation panel accordingly.
            """
            self.template.selection_panel._refresh_selected_app_metadata()

            if not any((self._work_dir / "Evaluation").rglob("*.json")):
                self._update_logs(
                    "[KonfAI] Evaluation finished but produced no metrics file "
                    f"(no .json under {self._work_dir / 'Evaluation'}). "
                    "The process probably failed: check the Python console for errors.",
                    False,
                )
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

        app = self.template.ui.appComboBox.currentData
        args = ["uncertainty", app.get_name(), "-i", "Volume.mha", "-o", "Uncertainty"]
        # The current tab's config, and its references in the manifest order: Reference_0, Reference_1...
        key = self._current_uncertainty_key()
        if key is not None:
            args += ["--uncertainty_file", key.evaluation_file]
            for name, (_required, selector) in self._uncertainty_input_selectors[key].items():
                if selector.currentNode() is not None:
                    args += ["--gt", self._save_entry(name, selector.currentNode())]

        # Device selection (GPU or CPU)
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += remote_server.cli_args()

        self.write_stack(self.ui.inputVolumeSequenceSelector.currentNode(), self._work_dir / "Volume.mha")

        def on_end_function() -> None:
            """
            Callback executed when the uncertainty process finishes.

            It loads metrics and MHA images produced by konfai-apps and updates
            the uncertainty panel accordingly.
            """
            self.template.selection_panel._refresh_selected_app_metadata()

            if not any((self._work_dir / "Uncertainty").rglob("*.json")):
                self._update_logs(
                    "[KonfAI] Uncertainty analysis finished but produced no metrics file "
                    f"(no .json under {self._work_dir / 'Uncertainty'}). "
                    "The process probably failed: check the Python console for errors.",
                    False,
                )
                return

            from konfai.evaluator import Statistics

            statistics = Statistics(next((self._work_dir / "Uncertainty").rglob("*.json")))
            self.uncertainty_panel.set_metrics(statistics.read())
            # A landmark workflow writes metrics alone: the list is then emptied, not left with the last run's maps.
            self.uncertainty_panel.refresh_images_list(self._work_dir / "Uncertainty")

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)
