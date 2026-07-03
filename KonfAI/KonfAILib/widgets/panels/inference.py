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

"""Inference panel: input/mask/output selection, checkpoint ensemble,
TTA/MC-dropout sampling options and the inference run itself."""

import random
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
from slicer.i18n import tr as _

from KonfAILib.logic.servers import RemoteServer
from KonfAILib.widgets import io_selectors
from KonfAILib.widgets.chip_selector import ChipSelector
from KonfAILib.widgets.panels.base import KonfAIAppPanel

if TYPE_CHECKING:
    from KonfAILib.widgets.app_template import KonfAIAppTemplateWidget


class KonfAIAppInferencePanel(KonfAIAppPanel):
    """
    Inference panel of a KonfAI application.

    This panel owns the input volume/mask and output volume selectors,
    the checkpoint chip selector (ensemble), the TTA/MC-dropout sampling
    controls, the uncertainty checkbox and the inference run button.
    """

    def __init__(self, template: "KonfAIAppTemplateWidget") -> None:
        super().__init__(template, "KonfAIAppInference")

        # Dynamic selectors for the extra inputs/outputs declared by the app
        # manifest (the first declared input/output stays on the static
        # selectors): {key: (required, qMRMLNodeComboBox)}.
        self._extra_input_selectors: dict = {}
        self._extra_output_selectors: dict = {}

        # Connect volume selectors to parameter node synchronization
        self.ui.inputVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.template.update_parameter_node_from_gui
        )
        self.ui.outputVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.template.update_parameter_node_from_gui
        )
        self.ui.outputVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.ui.segmentationShow3DButton.setSegmentationNode
        )
        self.ui.segmentationShow3DButton.setVisible(False)

        self.ui.ttaSpinBox.valueChanged.connect(self.on_tta_changed)
        self.ui.mcDropoutSpinBox.valueChanged.connect(self.on_mc_dropout_changed)
        # Advanced overrides (patch size per dim + batch size); None = auto (follow the app's VRAM plan).
        self._patch_default: list[int] | None = None
        self._patch_override: list[int] | None = None
        self._batch_override: int | None = None
        self.ui.advancedButton.clicked.connect(self.on_advanced_clicked)

        # Run button for inference
        self.ui.runInferenceButton.clicked.connect(self.on_run_inference_button)

        self.ui.uncertaintyCheckBox.toggled.connect(self.on_uncertainty_toggled)

        self.chip_selector = ChipSelector(
            self.ui.checkpointsComboBox,
            self.ui.selectedCheckpointsWidget.layout(),
            self.ui.ensembleSpinBox,
            1,
            on_change=self.on_checkpoint_selected_change,
        )

    def on_uncertainty_toggled(self, checked: bool) -> None:
        self.set_parameter("uncertainty", str(checked))

    def on_checkpoint_selected_change(self, checkpoints_selected: list[str]):
        self.set_parameter("checkpoints_name", ",".join(checkpoints_selected))

    def on_tta_changed(self):
        self.set_parameter("number_of_tta", str(self.ui.ttaSpinBox.value))

    def on_mc_dropout_changed(self):
        self.set_parameter("number_of_mc_dropout", str(self.ui.mcDropoutSpinBox.value))

    def on_advanced_clicked(self):
        """Open the advanced dialog: override the patch size (one spinbox per dimension) and batch size.

        The number of patch dimensions comes from the app (``get_patch_size()``), so a 2.5D app shows its
        ``[1, Y, X]`` dims and a 3D app shows ``[Z, Y, X]``. Leaving 'auto' checked follows the VRAM plan.
        """
        import qt

        patch = self._patch_override or self._patch_default or [96, 96, 96]
        dialog = qt.QDialog(self.ui.advancedButton)
        dialog.setWindowTitle("Advanced inference settings")
        form = qt.QFormLayout(dialog)

        auto_check = qt.QCheckBox("Follow the app's VRAM plan (auto)")
        auto_check.setChecked(self._patch_override is None and self._batch_override is None)
        form.addRow(auto_check)

        labels = ["Z", "Y", "X"] if len(patch) == 3 else [f"dim {i}" for i in range(len(patch))]
        spins = []
        for label, value in zip(labels, patch, strict=False):
            spin = qt.QSpinBox()
            spin.setRange(1, 4096)
            spin.setValue(int(value))
            form.addRow(f"Patch {label}:", spin)
            spins.append(spin)

        batch_spin = qt.QSpinBox()
        batch_spin.setRange(1, 4096)
        batch_spin.setValue(self._batch_override or 1)
        form.addRow("Batch size:", batch_spin)

        def _toggle(checked):
            for spin in spins:
                spin.setEnabled(not checked)
            batch_spin.setEnabled(not checked)

        auto_check.toggled.connect(_toggle)
        _toggle(auto_check.isChecked())

        buttons = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if not dialog.exec_():
            return
        if auto_check.isChecked():
            self._patch_override = None
            self._batch_override = None
        else:
            self._patch_override = [spin.value for spin in spins]
            self._batch_override = batch_spin.value
        self.set_parameter("patch_size", ",".join(str(v) for v in self._patch_override) if self._patch_override else "")
        self.set_parameter("batch_size", str(self._batch_override or ""))
        self._refresh_advanced_button()

    def _refresh_advanced_button(self):
        """Mark the gear when an override is active, so the user sees it is no longer on 'auto'."""
        if self._patch_override or self._batch_override:
            patch = "x".join(str(v) for v in self._patch_override) if self._patch_override else "auto"
            self.ui.advancedButton.setText("⚙*")
            self.ui.advancedButton.setToolTip(f"Override active — patch {patch}, batch {self._batch_override or 'auto'}.")
        else:
            self.ui.advancedButton.setText("⚙")
            self.ui.advancedButton.setToolTip("Advanced: override the patch size (per dimension) and batch size.")

    def on_app_changed(self, app) -> None:
        """
        Reconfigure the sampling controls when the selected app changes.

        Called by the template's ``dispatch_app_changed``.
        """
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

        # Advanced patch/batch override: seed the dialog with the app's own patch dims (2.5D vs 3D), and
        # reset any override to 'auto' so a per-app plan is followed unless the user opts in for this app.
        self._patch_default = app.get_patch_size() if hasattr(app, "get_patch_size") else None
        self._patch_override = None
        self._batch_override = None
        self._refresh_advanced_button()

        # Uncertainty estimation availability depends on app capabilities
        has_inference, has_evaluation, has_uncertainty = app.has_capabilities()
        self.ui.uncertaintyCheckBox.setEnabled(has_uncertainty)

        self._rebuild_io_selectors(app)

    def _rebuild_io_selectors(self, app) -> None:
        """
        Regenerate the dynamic selectors for the extra inputs/outputs declared by the app.

        The first declared input/output stays on the static selectors; apps
        without declarations keep the current static-only behavior (both
        dynamic forms are simply emptied).
        """
        inputs = dict(getattr(app, "get_inputs", dict)())
        outputs = dict(getattr(app, "get_outputs", dict)())

        self._extra_input_selectors = io_selectors.add_inputs_on_layout(
            self.ui.extraInputsForm.layout(), dict(list(inputs.items())[1:])
        )
        self._extra_output_selectors = io_selectors.add_outputs_on_layout(
            self.ui.extraOutputsForm.layout(), dict(list(outputs.items())[1:])
        )

        first_input = next(iter(inputs.values()), None)
        self.ui.label_input.setText(first_input.display_name if first_input else _("Input volume"))

        # Persist selections under Input/{key} / Output/{key} and restore them
        for prefix, selectors in (("Input", self._extra_input_selectors), ("Output", self._extra_output_selectors)):
            for key, (_required, selector) in selectors.items():
                selector.connect(
                    "currentNodeChanged(vtkMRMLNode*)",
                    lambda node, p=prefix, k=key: self._on_extra_io_changed(p, k, node),
                )
                selector.setCurrentNode(self.get_parameter_node(f"{prefix}/{key}"))

        self._default_distinct_volume_inputs()
        self._align_label_columns()

    def _default_distinct_volume_inputs(self) -> None:
        """Preselect a distinct volume for each still-empty volume input.

        A multi-input app (e.g. registration: fixed + moving) should not default both inputs to the same
        image. When several scalar volumes are loaded, fill each empty volume selector with the next unused
        one, in declaration order; selectors already set (restored from the parameter node) are untouched.
        """
        volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        used = {self.ui.inputVolumeSelector.currentNodeID}
        for _key, (_required, selector) in self._extra_input_selectors.items():
            if "vtkMRMLScalarVolumeNode" not in list(selector.nodeTypes):
                continue
            if selector.currentNode() is not None:
                used.add(selector.currentNodeID)
                continue
            nxt = next((v for v in volumes if v.GetID() not in used), None)
            if nxt is not None:
                selector.setCurrentNode(nxt)
                used.add(nxt.GetID())

    def _align_label_columns(self) -> None:
        """Align the value selectors of the three inference form layouts.

        ``formLayout_inputs`` and the nested ``extraInputsForm`` / ``extraOutputsForm`` forms each
        auto-size their label column to their own longest label, so their value selectors start at three
        different x positions. Feed every label (the four static ones plus the dynamic label-role widgets
        of both nested forms) to a shared-minimum-width pass so all label columns equalise and every
        selector lines up. Re-run on each rebuild; the pass resets widths first, so nothing accumulates.
        """
        labels = [
            self.ui.label_input,
            self.ui.label_stochastic,
            self.ui.label_selected_checkpoints,
            self.ui.label_output,
        ]
        labels += io_selectors.form_label_widgets(self.ui.extraInputsForm.layout())
        labels += io_selectors.form_label_widgets(self.ui.extraOutputsForm.layout())
        io_selectors.align_label_columns(labels)

    def _on_extra_io_changed(self, prefix: str, key: str, node) -> None:
        """Persist a dynamic selector change under its namespaced parameter node key."""
        if self.template._parameter_node is None or not self.template._initialized:
            return
        self.set_parameter_node(f"{prefix}/{key}", node.GetID() if node is not None else None)

    def required_inputs_ok(self) -> bool:
        """Return True when every required dynamic extra input has a usable node."""
        return all(
            not required or io_selectors.has_selection(selector)
            for required, selector in self._extra_input_selectors.values()
        )

    def _warn_on_input_gaps(self) -> None:
        """Warn when an empty optional input precedes a filled one in get_inputs() order.

        The CLI input mapping is positional: skipping an empty group shifts the
        following ones (Volume_{i} indices), so the app may misread them.
        """
        empty_key = None
        for key, (_required, selector) in self._extra_input_selectors.items():
            if selector.currentNode() is None:
                empty_key = empty_key or key
            elif empty_key is not None:
                self._update_logs(
                    f"[KonfAI] Warning: optional input '{empty_key}' is empty but '{key}' is set. "
                    "The CLI input mapping is positional, so the following input groups are "
                    "shifted and the app may misinterpret them.",
                    False,
                )
                return

    def _export_extra_inputs(self) -> list[str]:
        """Export the dynamic extra inputs by key and return their CLI '-i' flags."""
        args: list[str] = []
        for key, (_required, selector) in self._extra_input_selectors.items():
            node = selector.currentNode()
            if node is None:
                continue
            if node.IsA("vtkMRMLTransformNode"):
                file_name = f"{key}.h5"
                slicer.util.saveNode(node, str(self._work_dir / file_name))
            elif node.IsA("vtkMRMLMarkupsNode"):
                file_name = f"{key}.fcsv"
                slicer.util.saveNode(node, str(self._work_dir / file_name))
            else:
                file_name = f"{key}.mha"
                sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(node), str(self._work_dir / file_name))
            self._update_logs(f"Extra input saved to temporary folder: {self._work_dir / file_name}")
            args += ["-i", file_name]
        return args

    def on_run_inference_button(self) -> None:
        """
        Run or stop inference depending on the current state.
        """
        self.on_run_button(self.inference)

    def inference(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """
        Run inference using the selected KonfAI app.

        Steps:
          - Export the input volume to an MHA file with appropriate metadata
          - Call `konfai-apps infer` with ensemble/TTA/MC-dropout options
          - Load the resulting MHA files back into Slicer and update display
          - Populate uncertainty panel with the generated stack if available
        """
        if remote_server is not None and self._extra_input_selectors:
            self._update_logs(
                "[KonfAI] Multi-input apps are not supported on remote servers yet "
                "(konfai-apps flattens input groups); run locally.",
                False,
            )
            self.template.set_running(False)
            return

        self.template.evaluation_panel.clear_images_list()
        self.template.uncertainty_panel.clear_images_list()
        self.template.evaluation_panel.clear_metrics()
        self.template.uncertainty_panel.clear_metrics()

        app = self.template.ui.appComboBox.currentData
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
        # Advanced overrides: patch size (per dimension) + batch size. Unset = auto (follow the VRAM plan).
        if self._patch_override:
            args += ["--patch-size", *[str(v) for v in self._patch_override]]
        if self._batch_override:
            args += ["--batch-size", str(self._batch_override)]
        if self.ui.uncertaintyCheckBox.isChecked():
            args += ["-uncertainty"]

        # First local run: selected checkpoints may not be on disk yet.
        # --download makes konfai-apps fetch the app upfront, with its
        # progress visible in the log panel instead of a silent stall.
        if remote_server is None:
            missing = set(self.chip_selector.selected()) - set(app.get_checkpoints_name_available())
            if missing:
                args += ["--download"]
                self._update_logs(
                    f"[KonfAI] Downloading missing model file(s) before the run: {sorted(missing)}", False
                )

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

        # Extra '-i' groups keep the get_inputs() order: the main static input
        # ('-i Volume.mha' above) is the first declared entry (Volume_0).
        self._warn_on_input_gaps()
        args += self._export_extra_inputs()

        def on_end_function() -> None:
            """
            Callback executed when inference finishes.

            It reads the first non-stack output file, loads it into Slicer, sets
            appropriate volume class (scalar vs labelmap), copies orientation and
            attributes, and configures slice viewer overlays.
            """
            refreshed_app = self.template.selection_panel._refresh_selected_app_metadata()
            if refreshed_app is not None:
                app = refreshed_app
            else:
                app = self.template.ui.appComboBox.currentData

            data = None
            # Find the first non-stack MHA output file
            for file in (self._work_dir / "Output").rglob("*.mha"):
                if file.name != "InferenceStack.mha":
                    from konfai.utils.dataset import image_to_data

                    data, attr = image_to_data(sitk.ReadImage(str(file)))
                    break
            if data is None:
                return

            # Load the additional output files into the dynamic selectors by
            # matching each declared output key to the file whose path contains
            # it (rglob order is not deterministic); the mono-output path above
            # stays untouched.
            if self._extra_output_selectors:
                files = [f for f in (self._work_dir / "Output").rglob("*.mha") if f.name != "InferenceStack.mha"]

                def match_output_file(output_key: str):
                    return next((f for f in files if output_key in f.parts or f.stem == output_key), None)

                # The main output is the file matched to the first key of
                # get_outputs(); when unmatched, keep the first file found above.
                main_key = next(iter(dict(getattr(app, "get_outputs", dict)())), None)
                main_file = match_output_file(main_key) if main_key else None
                if main_file is not None and main_file != file:
                    from konfai.utils.dataset import image_to_data

                    file = main_file
                    data, attr = image_to_data(sitk.ReadImage(str(file)))

                for key, (required, selector) in self._extra_output_selectors.items():
                    extra_file = match_output_file(key)
                    if extra_file is None:
                        if required:
                            self._update_logs(
                                f"[KonfAI] Required output '{key}' has no matching file under "
                                f"{self._work_dir / 'Output'}; it cannot be loaded into Slicer.",
                                False,
                            )
                        continue
                    image = sitk.ReadImage(str(extra_file))
                    if "vtkMRMLSegmentationNode" in list(selector.nodeTypes):
                        segmentation_node = selector.currentNode()
                        if segmentation_node is None:
                            segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
                                "vtkMRMLSegmentationNode", f"{key}_{app.get_name()}"
                            )
                            selector.setCurrentNode(segmentation_node)
                        tmp_extra_labelmap = sitkUtils.PushVolumeToSlicer(
                            image, name=f"{key}_LabelMap", className="vtkMRMLLabelMapVolumeNode"
                        )
                        segmentation_node.CreateDefaultDisplayNodes()
                        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                            tmp_extra_labelmap, segmentation_node
                        )
                        slicer.mrmlScene.RemoveNode(tmp_extra_labelmap)
                    else:
                        extra_node = sitkUtils.PushVolumeToSlicer(image, targetNode=selector.currentNode(), name=key)
                        selector.setCurrentNode(extra_node)

            self._update_logs("Loading result into Slicer...")

            # Decide if output should be a label map (uint8) or scalar volume
            want_label = data.dtype == np.uint8
            expected_class = "vtkMRMLSegmentationNode" if want_label else "vtkMRMLScalarVolumeNode"
            base_name = self.ui.outputVolumeSelector.baseName + "_" + app.get_name()

            node = slicer.mrmlScene.AddNewNodeByClass(
                expected_class, base_name + ("_Segmentation" if want_label else "_Output")
            )
            self.ui.outputVolumeSelector.setCurrentNode(node)
            self.ui.segmentationShow3DButton.setVisible(node.IsA("vtkMRMLSegmentationNode"))
            if node.IsA("vtkMRMLSegmentationNode"):
                tmp_labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", base_name + "_LabelMap")
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

            sequence_node = self.template.ui.inputVolumeSequenceSelector.currentNode()
            if sequence_node is None:
                sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "InputTransformSequence")
                self.template.ui.inputVolumeSequenceSelector.setCurrentNode(sequence_node)
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
