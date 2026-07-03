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
import sitkUtils
import slicer
from qt import QFormLayout, QTabWidget, QVBoxLayout, QWidget

from KonfAILib.logic.mrml import has_node_content
from KonfAILib.logic.servers import RemoteServer
from KonfAILib.widgets import io_selectors
from KonfAILib.widgets.metrics_panel import KonfAIMetricsPanel
from KonfAILib.widgets.panels.base import KonfAIAppPanel

if TYPE_CHECKING:
    from KonfAILib.widgets.app_template import KonfAIAppTemplateWidget


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

        # Internal tabs: one per EvaluationKey declared by the app manifest;
        # hidden in static (single evaluation) mode.
        self._evaluation_tabs = QTabWidget()
        self._evaluation_tabs.setVisible(False)
        self.ui.evaluationTabsPlaceholder.layout().addWidget(self._evaluation_tabs)
        self._evaluation_tabs.currentChanged.connect(self.on_tab_changed)
        # {EvaluationKey: {key: (required, qMRMLNodeComboBox)}} in manifest order
        self._evaluation_input_selectors: dict = {}

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

    def _rebuild_evaluation_tabs(self, app) -> None:
        """
        Regenerate one internal tab of dynamic selectors per declared evaluation.

        Apps without declared evaluations keep the static selector rows
        (current behavior); otherwise the static input/reference rows are hidden.
        """
        evaluations_inputs = dict(getattr(app, "get_evaluations_inputs", dict)())

        # QTabWidget.clear() does not destroy the pages: delete them explicitly
        # so their qMRMLNodeComboBox children stop observing the scene.
        for i in range(self._evaluation_tabs.count):
            self._evaluation_tabs.widget(i).deleteLater()
        self._evaluation_tabs.clear()
        self._evaluation_input_selectors = {}

        # The positional CLI routing needs at least an input and a reference
        # entry per evaluation; otherwise keep the full static fallback.
        dynamic = bool(evaluations_inputs) and all(len(inputs) >= 2 for inputs in evaluations_inputs.values())
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

        for evaluation_key, evaluation_inputs in evaluations_inputs.items():
            tab = QWidget()
            root = QVBoxLayout(tab)
            root.setContentsMargins(12, 12, 12, 12)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            selectors = io_selectors.add_inputs_on_layout(form, evaluation_inputs)
            root.addLayout(form)
            root.addStretch(1)
            self._evaluation_tabs.addTab(tab, evaluation_key.display_name)
            self._evaluation_input_selectors[evaluation_key] = selectors

            # Persist selections under Eval/{evaluation_file}/{key} and restore them
            for key, (_required, selector) in selectors.items():
                param_key = f"Eval/{evaluation_key.evaluation_file}/{key}"
                selector.connect(
                    "currentNodeChanged(vtkMRMLNode*)",
                    lambda node, pk=param_key: self._on_evaluation_selector_changed(pk, node),
                )
                selector.setCurrentNode(self.get_parameter_node(param_key))

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
            # current internal tab (1st entry -> -i, 2nd entry -> --gt, then
            # Mask* -> --mask, anything else -> an extra -i group).
            items = list(self._evaluation_input_selectors[evaluation_key].items())
            selectors = dict(items)
            # The resampling reference is the node routed to --gt (2nd entry)
            _gt_required, gt_selector = items[1][1]
            reference_node = gt_selector.currentNode()
            if reference_node is not None and not reference_node.IsA("vtkMRMLVolumeNode"):
                reference_node = None
            args += ["-o", "Evaluation"]
            for position, (key, (_required, selector)) in enumerate(items):
                node = selector.currentNode()
                if node is None:
                    continue
                if position == 0:
                    flag = "-i"
                elif position == 1:
                    flag = "--gt"
                else:
                    flag = "--mask" if key.startswith("Mask") else "-i"
                if node.IsA("vtkMRMLTransformNode") or node.IsA("vtkMRMLMarkupsNode"):
                    file_name = key + (".h5" if node.IsA("vtkMRMLTransformNode") else ".fcsv")
                    slicer.util.saveNode(node, str(self._work_dir / file_name))
                else:
                    file_name = f"{key}.mha"
                    if flag == "--gt" or reference_node is None:
                        self._write_volume(node, self._work_dir / file_name)
                    else:
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
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

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
            self.uncertainty_panel.refresh_images_list(
                Path(next((self._work_dir / "Uncertainty").rglob("*.mha")).parent)
            )

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)
