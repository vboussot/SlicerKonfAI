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

"""Dynamic qMRML node selectors generated from a KonfAI app manifest.

The konfai-apps ``DataEntry`` objects are handled through duck-typing only
(``entry.volume_type.value`` string comparison), so this module never
imports konfai_apps.
"""

import slicer
from qt import QFormLayout, QHBoxLayout, QLabel, QSizePolicy, Qt, QWidget

from KonfAILib.logic.mrml import has_node_content

# Contextual VolumeType -> nodeTypes mappings. SEGMENTATION maps to labelmaps
# on the input/evaluation side (segmentation nodes are only produced as
# inference outputs) so vtkMRMLSegmentationNode never appears in input selectors.
_INPUT_NODE_TYPES = {
    "VOLUME": ["vtkMRMLScalarVolumeNode"],
    "SEGMENTATION": ["vtkMRMLLabelMapVolumeNode"],
    "FIDUCIALS": ["vtkMRMLMarkupsFiducialNode"],
    "TRANSFORM": ["vtkMRMLTransformNode"],
}
_OUTPUT_NODE_TYPES = dict(_INPUT_NODE_TYPES, SEGMENTATION=["vtkMRMLSegmentationNode"])


def volume_type_value(entry) -> str:
    """Return the VolumeType of a konfai-apps DataEntry as a plain string."""
    return str(getattr(entry.volume_type, "value", entry.volume_type))


def has_selection(selector) -> bool:
    """Return True when the selector holds a usable node (with content for image nodes)."""
    node = selector.currentNode()
    if node is None:
        return False
    if node.IsA("vtkMRMLVolumeNode") or node.IsA("vtkMRMLSegmentationNode"):
        return has_node_content(node)
    return True


def clear_form_layout(layout: QFormLayout) -> None:
    while layout.rowCount() > 0:
        layout.removeRow(0)


def form_label_widgets(layout: QFormLayout) -> list[QLabel]:
    """Return the label-role widgets currently in a form layout, in row order."""
    labels: list[QLabel] = []
    for row in range(layout.rowCount()):
        item = layout.itemAt(row, QFormLayout.LabelRole)
        widget = item.widget() if item is not None else None
        if widget is not None:
            labels.append(widget)
    return labels


def align_label_columns(labels: list) -> None:
    """Give every label the widest label's preferred width as its minimum width.

    Equalising the minimum width forces the label column of each ``QFormLayout``
    to the same size, so the value selectors of every form start at the same x.
    Minimum widths are reset first so the column never accumulates across
    successive app changes (``sizeHint`` is the natural, min-independent width).
    """
    for label in labels:
        label.setMinimumWidth(0)
    widths = [label.sizeHint.width() for label in labels]
    if not widths:
        return
    shared = max(widths)
    for label in labels:
        label.setMinimumWidth(shared)


def _make_row(layout: QFormLayout, entry, node_types: list[str]):
    label = QLabel(entry.display_name)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    selector = slicer.qMRMLNodeComboBox()
    selector.setMRMLScene(slicer.mrmlScene)
    selector.nodeTypes = node_types
    selector.addEnabled = False
    selector.removeEnabled = False
    selector.showChildNodeTypes = False
    selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    selector.setFixedHeight(26)
    return label, selector


def add_inputs_on_layout(layout: QFormLayout, inputs: dict) -> dict:
    """Fill a form layout with one input selector per DataEntry.

    Returns ``{key: (required, selector)}`` in ``inputs`` insertion order.
    """
    selectors = {}
    clear_form_layout(layout)
    for key, entry in inputs.items():
        node_types = _INPUT_NODE_TYPES.get(volume_type_value(entry), ["vtkMRMLScalarVolumeNode"])
        label, selector = _make_row(layout, entry, node_types)
        selector.noneEnabled = not entry.required
        layout.addRow(label, selector)
        selectors[key] = (entry.required, selector)
    return selectors


def add_outputs_on_layout(layout: QFormLayout, outputs: dict) -> dict:
    """Fill a form layout with one output selector per DataEntry.

    Returns ``{key: (required, selector)}`` in ``outputs`` insertion order.
    """
    selectors = {}
    clear_form_layout(layout)
    for key, entry in outputs.items():
        node_types = _OUTPUT_NODE_TYPES.get(volume_type_value(entry), ["vtkMRMLScalarVolumeNode"])
        label, selector = _make_row(layout, entry, node_types)
        selector.noneEnabled = True
        selector.addEnabled = True
        selector.removeEnabled = True
        selector.editEnabled = True
        selector.renameEnabled = True
        selector.noneDisplay = "Create new volume on Run"
        selector.baseName = key

        if volume_type_value(entry) == "SEGMENTATION":
            # Composite row: selector + show-3D button, as for the main output
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            show_3d_button = slicer.qMRMLSegmentationShow3DButton()
            selector.connect("currentNodeChanged(vtkMRMLNode*)", show_3d_button.setSegmentationNode)
            row_layout.addWidget(selector, 1)
            row_layout.addWidget(show_3d_button)
            layout.addRow(label, row_widget)
        else:
            layout.addRow(label, selector)
        selectors[key] = (entry.required, selector)
    return selectors
