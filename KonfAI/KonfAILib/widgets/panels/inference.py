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
from pathlib import Path
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
        # setSegmentationNode() rejects non-segmentation nodes; the output selector also holds scalar
        # volumes (registration / synthesis outputs), so route through a type guard instead of connecting
        # setSegmentationNode directly (which would raise TypeError on a scalar-volume output).
        self.ui.outputVolumeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._sync_segmentation_show3d
        )
        self.ui.segmentationShow3DButton.setVisible(False)

        self.ui.ttaSpinBox.valueChanged.connect(self.on_tta_changed)
        self.ui.mcDropoutSpinBox.valueChanged.connect(self.on_mc_dropout_changed)
        # Advanced overrides (patch size per dim + batch size); None = auto (follow the app's VRAM plan).
        self._patch_default: list[int] | None = None
        self._patch_override: list[int] | None = None
        self._batch_override: int | None = None
        # Tuned model parameters {config key: value}; empty = the app's own config defaults. A nested value
        # (e.g. the whole `resolutions` matrix) is just one entry here — it gets no special-casing.
        self._param_override: dict[str, object] = {}
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

    def _current_free_vram(self) -> float | None:
        """Free VRAM (GB) for the selected GPU(s). The measurement itself lives in konfai-apps
        (``current_free_vram`` — the same logic inference uses to pick its VRAM plan); this method only
        gathers the UI state (selected devices, remote server) and delegates."""
        try:
            devices = self.template.get_device()
        except Exception:
            return None
        if not devices:
            return None
        try:
            remote_server, _ = self.get_remote_server()
        except Exception:
            remote_server = None
        from konfai_apps.app_repository import current_free_vram

        return current_free_vram([int(d) for d in devices], remote_server)

    def _sync_segmentation_show3d(self, node) -> None:
        """Forward only segmentation nodes to the Show-3D button; a scalar-volume output would otherwise
        raise 'method requires a vtkMRMLSegmentationNode, a vtkMRMLScalarVolumeNode was provided'."""
        seg = node if (node is not None and node.IsA("vtkMRMLSegmentationNode")) else None
        self.ui.segmentationShow3DButton.setSegmentationNode(seg)

    def on_advanced_clicked(self):
        """Open the advanced dialog: override the patch size (one spinbox per dimension) and batch size.

        The spinboxes are seeded with the plan konfai-apps would actually run on this machine now: the
        app's VRAM plan resolved for the selected device's free VRAM (which carries a patch geometry even
        when the app declares no static ``patch_size``). A 2.5D app shows ``[1, Y, X]`` (first axis locked
        to 1); leaving 'auto' checked follows the plan without an override.
        """
        import qt

        app = self.template.ui.appComboBox.currentData
        plan_patch, plan_batch = None, None
        if app is not None and hasattr(app, "resolve_vram_plan"):
            resolved = app.resolve_vram_plan(self._current_free_vram())
            if resolved:
                plan_patch, plan_batch = resolved

        patch = self._patch_override or plan_patch or self._patch_default
        default_batch = self._batch_override or plan_batch or 1
        dialog = qt.QDialog(self.ui.advancedButton)
        dialog.setWindowTitle("Advanced inference settings")
        # Roomy + scrollable: the model-parameter matrix can be tall, so wrap the form in a scroll area
        # and pin the OK/Cancel box below it. A generous minimum size opens the dialog wide enough to read
        # the full title and the matrix grid without hand-resizing.
        dialog.setMinimumSize(600, 520)
        outer = qt.QVBoxLayout(dialog)
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qt.QFrame.NoFrame)
        content = qt.QWidget()
        form = qt.QFormLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        auto_check = qt.QCheckBox("Follow the app's VRAM plan (auto)")
        auto_check.setChecked(self._patch_override is None and self._batch_override is None)
        form.addRow(auto_check)

        spins = []
        if patch:
            labels = ["Z", "Y", "X"] if len(patch) == 3 else [f"dim {i}" for i in range(len(patch))]
            for i, (label, value) in enumerate(zip(labels, patch, strict=False)):
                spin = qt.QSpinBox()
                if int(value) == 1:
                    # A patch dimension of 1 is fixed (slice-wise on that axis) — don't offer it. Keep a
                    # hidden value holder (never shown) so the override still carries the full patch.
                    spin.setRange(1, 1)
                    spin.setValue(1)
                    spins.append(spin)
                    continue
                spin.setRange(1, 4096)
                spin.setValue(int(value))
                form.addRow(f"Patch {label}:", spin)
                spins.append(spin)
        else:
            # No declared patch geometry: overriding a patch we don't know would clash with the app's
            # extend_slice; only expose the batch override here.
            note = qt.QLabel("This app does not declare a patch size, so only the batch size can be overridden.")
            note.setWordWrap(True)
            form.addRow(note)

        batch_spin = qt.QSpinBox()
        batch_spin.setRange(1, 4096)
        batch_spin.setValue(default_batch)
        form.addRow("Batch size:", batch_spin)

        def _toggle(checked):
            for spin in spins:
                spin.setEnabled(not checked)
            batch_spin.setEnabled(not checked)

        auto_check.toggled.connect(_toggle)
        _toggle(auto_check.isChecked())

        # Model parameters: the configurable knobs + their constraints, read in ONE generic call. The app's
        # `get_parameters()` returns {values: <nested dict from the resolved config>, constraints: <parallel
        # sparse tree from the model TYPES>}. Each value is rendered purely by STRUCTURE, overlaid with its
        # constraint (choices -> dropdown, {min,max} -> spin bounds). A nested value such as the per-resolution
        # `resolutions` matrix is just one entry, edited by the same recursive editor — no special case. No
        # app.json declaration is needed; "Save as local app" bakes the current values into a local copy.
        try:
            params = app.get_parameters() if (app is not None and hasattr(app, "get_parameters")) else {}
        except Exception:  # noqa: BLE001 - a config/type read failure must not break the dialog
            params = {}
        values = params.get("values") or {}
        constraints = params.get("constraints") or {}
        param_widgets: list[tuple[str, object, object]] = []
        if values:
            form.addRow(qt.QLabel("<b>Model parameters</b>"))
            for name, default in values.items():
                current = self._param_override.get(name, default)
                widget, reader = self._build_value_editor(current, constraints.get(name), name)
                param_widgets.append((name, default, reader))
                form.addRow(f"{name}:", widget)

            def _on_save():
                self._save_as_local_app(app, param_widgets, dialog)

            save_button = qt.QPushButton("Save as local app…")
            save_button.setToolTip("Copy this app into a local folder with the current parameters as its defaults.")
            save_button.clicked.connect(_on_save)
            form.addRow(save_button)

        # Add the buttons explicitly: in PythonQt, QDialogButtonBox(Ok | Cancel) can mis-resolve the
        # OR'ed int to the (orientation) constructor overload and render a box with NO buttons.
        buttons = qt.QDialogButtonBox()
        buttons.addButton(qt.QDialogButtonBox.Ok)
        buttons.addButton(qt.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)  # pinned below the scroll area, always visible even with a tall matrix

        if not dialog.exec_():
            return
        if auto_check.isChecked():
            self._patch_override = None
            self._batch_override = None
        else:
            self._patch_override = [spin.value for spin in spins]
            self._batch_override = batch_spin.value
        # Model-parameter overrides apply regardless of the patch/batch 'auto' toggle.
        self._param_override = self._collect_param_overrides(param_widgets)
        self._refresh_advanced_button()

    def _refresh_advanced_button(self):
        """Mark the gear when an override is active, so the user sees it is no longer on 'auto'."""
        if self._patch_override or self._batch_override or self._param_override:
            patch = "x".join(str(v) for v in self._patch_override) if self._patch_override else "auto"
            tip = f"Override active — patch {patch}, batch {self._batch_override or 'auto'}"
            if self._param_override:
                tip += f", {len(self._param_override)} parameter(s)"
            self.ui.advancedButton.setText("⚙*")
            self.ui.advancedButton.setToolTip(tip + ".")
        else:
            self.ui.advancedButton.setText("⚙")
            self.ui.advancedButton.setToolTip("Advanced: override patch size, batch size, and model parameters.")

    @staticmethod
    def _format_config_value(value) -> str:
        """Serialise a tuned value for ``--set name=value`` (konfai-apps parses it back as YAML).

        Strings are always double-quoted so values with YAML-significant characters survive round-trip —
        e.g. a feature-model ref ``VBoussot/impact-torchscript-models:MIND/R1D2_3D.pt`` (the ``:`` would
        otherwise read as a mapping) — and lists quote each string element the same way.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(value, list):
            return "[" + ", ".join(KonfAIAppInferencePanel._format_config_value(item) for item in value) + "]"
        if isinstance(value, dict):
            # A nested mapping (e.g. the whole `resolutions` matrix) as a YAML flow map; keys are quoted too.
            body = ", ".join(
                f"{KonfAIAppInferencePanel._format_config_value(k)}: {KonfAIAppInferencePanel._format_config_value(v)}"
                for k, v in value.items()
            )
            return "{" + body + "}"
        return str(value)

    def _build_value_editor(self, value, constraint=None, name: str = ""):
        """Generic RECURSIVE editor for one exposed parameter value — returns ``(widget, read_fn)``.

        SlicerKonfAI interprets a parameter by STRUCTURE, overlaid with the app's ``constraint`` (a sparse tree
        that mirrors the value): a scalar is a typed widget — a dropdown when the constraint carries
        ``{"choices": [...]}``, a spinbox bounded by ``{"min", "max"}``; a list of scalars an inline comma
        list; a nested object a group of rows; and a KonfAI dict-of-objects (keys ``'0','1',...``, as the
        constraint marks with a ``"*"`` wildcard) a repeatable block with add/remove. The ``resolutions``
        matrix is therefore not special — just a value whose type is "dict of objects". ``read_fn()`` returns
        the value in KonfAI config form: a repeatable block round-trips as an indexed ``{'0': {...}}`` mapping
        (what ``--set`` expects). No field name and no choice list is ever hard-coded here.
        """
        import copy

        import qt

        constraint = constraint or {}
        container = qt.QWidget()
        outer = qt.QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        state = {"root": copy.deepcopy(value), "read": lambda: copy.deepcopy(value)}

        def is_object_list(candidate):
            return isinstance(candidate, list) and bool(candidate) and all(isinstance(i, dict) for i in candidate)

        def is_object_map(candidate):  # a KonfAI dict[str, Object]: numeric string keys, object values
            return (
                isinstance(candidate, dict)
                and bool(candidate)
                and all(str(k).lstrip("-").isdigit() for k in candidate)
                and all(isinstance(v, dict) for v in candidate.values())
            )

        def is_repeatable(candidate, cst):
            # A repeatable collection: the constraint marks it with ``"*"``, or it structurally looks like a
            # KonfAI dict-of-objects / list-of-objects (so it still renders even without a constraint).
            return "*" in (cst or {}) or is_object_list(candidate) or is_object_map(candidate)

        def items_of(candidate):  # normalise a dict-of-objects (or list) to an ordered list for editing
            if isinstance(candidate, dict):
                return [candidate[k] for k in sorted(candidate, key=lambda k: int(k))]
            return list(candidate) if isinstance(candidate, list) else []

        def to_output(node):
            if is_object_list(node):
                return {str(i): to_output(item) for i, item in enumerate(node)}
            if isinstance(node, dict):
                return {key: to_output(val) for key, val in node.items()}
            return node

        def navigate(root, path):
            node = root
            for key in path:
                node = node[key]
            return node

        def scalar_widget(v, cst):
            cst = cst or {}
            choices = cst.get("choices")
            if choices is not None and not isinstance(v, (list, dict)):
                # A field with declared choices becomes an editable dropdown (editable, so a value outside the
                # list — e.g. a local model ref — is still allowed). The values come from the app.
                combo = qt.QComboBox()
                combo.setEditable(True)
                combo.addItems([str(option) for option in choices])
                combo.setCurrentText(str(v))
                return combo, (lambda: combo.currentText)
            if isinstance(v, bool):  # bool before int (bool is a subclass of int)
                w = qt.QCheckBox()
                w.setChecked(v)
                return w, (lambda: bool(w.isChecked()))
            if isinstance(v, int):
                w = qt.QSpinBox()
                lo = int(cst["min"]) if "min" in cst else -2147483648
                hi = int(cst["max"]) if "max" in cst else 2147483647
                w.setRange(max(lo, -2147483648), min(hi, 2147483647))
                w.setValue(v)
                return w, (lambda: int(w.value))
            if isinstance(v, float):
                w = qt.QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(float(cst.get("min", -1e12)), float(cst.get("max", 1e12)))
                w.setValue(v)
                return w, (lambda: float(w.value))
            if isinstance(v, str):
                w = qt.QLineEdit(str(v))
                return w, (lambda: w.text)
            # list of scalars (or empty): comma-separated, element type inferred for the round-trip
            as_list = v if isinstance(v, list) else []
            bools = any(isinstance(item, bool) for item in as_list)
            floats = any(isinstance(item, float) for item in as_list)
            strings = any(isinstance(item, str) for item in as_list)
            w = qt.QLineEdit(", ".join(str(item) for item in as_list))
            w.setToolTip("Comma-separated values (leave empty for none).")

            def read_list():
                items = [piece.strip() for piece in w.text.split(",") if piece.strip()]
                if bools:
                    return [item.lower() in ("true", "1", "yes") for item in items]
                if strings:
                    return items
                try:
                    return [float(item) if floats else int(item) for item in items]
                except ValueError:
                    return [int(i) if i.lstrip("-").isdigit() else float(i) for i in items]

            return w, read_list

        def render_object(obj, cst, path):
            cst = cst or {}
            box = qt.QWidget()
            form = qt.QFormLayout(box)
            form.setContentsMargins(0, 0, 0, 0)
            readers = {}
            for key, val in obj.items():
                subwidget, subread = render(val, cst.get(key), key, path + [key])
                form.addRow(f"{key}:", subwidget)
                readers[key] = subread
            return box, (lambda: {key: reader() for key, reader in readers.items()})

        def render_object_list(items, item_cst, field, path):
            # Structural heuristic (no field-name knowledge): objects that themselves hold a nested collection
            # stack VERTICALLY (they are big — e.g. resolutions holding models); leaf objects (scalars only) go
            # SIDE BY SIDE, so a resolution's models stay compact.
            nested = any(is_object_list(val) or is_object_map(val) for item in items for val in item.values())
            box = qt.QWidget()
            layout = qt.QVBoxLayout(box) if nested else qt.QHBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            readers = []
            for index, item in enumerate(items):
                cell = qt.QGroupBox(f"{field} {index}")
                if not nested:
                    cell.setMinimumWidth(240)
                cell_layout = qt.QVBoxLayout(cell)
                subwidget, subread = render_object(item, item_cst, path + [index])
                cell_layout.addWidget(subwidget)
                remove = qt.QPushButton("✕ remove")
                remove.clicked.connect(lambda _=0, p=path, i=index: drop(p, i))
                cell_layout.addWidget(remove)
                layout.addWidget(cell)
                readers.append(subread)
            add = qt.QPushButton("+ add")
            add.clicked.connect(lambda _=0, p=path: append(p))
            layout.addWidget(add)
            if not nested:
                layout.addStretch(1)
            return box, (lambda: [reader() for reader in readers])

        def render(v, cst, field, path):
            cst = cst or {}
            if is_repeatable(v, cst):
                return render_object_list(items_of(v), cst.get("*", {}), field, path)
            if isinstance(v, dict):
                return render_object(v, cst, path)
            return scalar_widget(v, cst)

        def append(path, empty=None):
            state["root"] = state["read"]()
            target = navigate(state["root"], path)
            target.append(copy.deepcopy(target[-1]) if target else ({} if empty is None else empty))
            rebuild()

        def drop(path, index):
            state["root"] = state["read"]()
            target = navigate(state["root"], path)
            if len(target) > 1:
                del target[index]
                rebuild()

        def rebuild():
            while outer.count():
                item = outer.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            widget, read = render(state["root"], constraint, name, [])
            state["read"] = read
            outer.addWidget(widget)

        rebuild()
        return container, (lambda: to_output(state["read"]()))

    def _collect_param_overrides(self, param_widgets) -> dict:
        """Keep only the parameters whose edited value differs from the config default."""
        overrides: dict = {}
        for name, default, reader in param_widgets:
            value = reader()
            if value is not None and value != default:
                overrides[name] = value
        return overrides

    def _param_override_set_args(self) -> list[str]:
        """The active model-parameter overrides (scalars and nested values alike) as ``--set`` CLI args."""
        args: list[str] = []
        for name, value in self._param_override.items():
            args += ["--set", f"{name}={self._format_config_value(value)}"]
        return args

    def _save_as_local_app(self, app, param_widgets, dialog) -> None:
        """Export the app into a local folder with the current parameters baked in as defaults (à la fine-tune)."""
        import re

        import qt
        from konfai_apps.app_repository import LocalAppRepositoryFromDirectory

        overrides = self._collect_param_overrides(param_widgets)
        set_args = [f"{name}={self._format_config_value(value)}" for name, value in overrides.items()]

        parent_dir = qt.QFileDialog.getExistingDirectory(None, "Choose parent directory for the new local app")
        if not parent_dir:
            return
        name = qt.QInputDialog.getText(None, "Save as local app", "New app name (folder will be created):")
        if not name or not name.strip():
            return
        display_name = qt.QInputDialog.getText(None, "Save as local app", "New display name:")
        if not display_name or not display_name.strip():
            return
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        if not safe:
            qt.QMessageBox.warning(None, "Invalid name", "Please enter a valid name.")
            return

        selection = self.template.selection_panel
        listed = [selection.ui.appComboBox.itemText(i) for i in range(selection.ui.appComboBox.count)]
        if display_name.strip() in listed:
            qt.QMessageBox.critical(None, "App already listed", f'The app "{display_name.strip()}" is already listed.')
            return

        try:
            app.export_app(Path(parent_dir) / safe, display_name=display_name.strip(), config_overrides=set_args)
        except Exception as exc:  # noqa: BLE001 - surface any export/copy failure to the user
            qt.QMessageBox.critical(None, "Save as local app", f"Could not save the app:\n{exc}")
            return

        new_app = LocalAppRepositoryFromDirectory(Path(parent_dir), safe)
        selection.ui.appComboBox.addItem(new_app.get_display_name(), new_app)
        selection.ui.appComboBox.setCurrentIndex(selection.ui.appComboBox.findData(new_app))
        selection.app_local_repositoy.append(new_app.get_name())
        self._update_logs(f"[KonfAI] Saved local app '{display_name.strip()}' at {Path(parent_dir) / safe}.")
        dialog.accept()

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

        # Reset any override when the app changes: a patch/batch chosen for one model must not carry over
        # to another (different geometry / VRAM plan). The new app starts fresh from its own plan.
        self._patch_default = app.get_patch_size() if hasattr(app, "get_patch_size") else None
        self._patch_override = None
        self._batch_override = None
        self._param_override = {}
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
        # Tuned model parameters (from the Advanced dialog) as --set path=value.
        args += self._param_override_set_args()
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
