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

"""The Inspect tab of the QA panel: KonfAI Scout reads a result the way a person would at a viewer and lists what
it finds, each finding one click away in the slice views.

A prototype: the inspector is chosen in the tab itself (it belongs beside the remote server), no reference yet."""

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import qt
import slicer

from KonfAILib.logic.mrml import has_node_content

if TYPE_CHECKING:
    from KonfAILib.widgets.panels.qa import KonfAIAppQAPanel

#: Who reads the result: the name shown, Scout's provider, whether the images leave the machine.
INSPECTORS = (
    ("Heuristic (no model)", "mock", False),
    ("Local server", "openai", False),
    ("Claude (your Claude Code login)", "claude-code", True),
)
SCALAR = ["vtkMRMLScalarVolumeNode"]
LABELS = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
#: Of each task, Scout's roles for the two volumes, their names in the form and the nodes the second may be.
TASKS = {
    "segmentation": (("image", "Image"), ("prediction", "Result"), LABELS),
    "registration": (("fixed", "Fixed image"), ("moving", "Moving image"), SCALAR),
    "synthesis": (("source", "Source image"), ("synthetic", "Synthetic image"), SCALAR),
}
#: How hard the inspector looks: Scout's depths, by the tool calls they allow and the effort they ask.
DEPTHS = (
    ("quick", "Quick: 10 looks, low effort"),
    ("standard", "Standard: 25 looks, high effort"),
    ("deep", "Deep: 40 looks, and each error re-read blind"),
)
SETTINGS = "KonfAI-Settings/Inspect/"


class InspectTab(qt.QWidget):
    def __init__(self, panel: "KonfAIAppQAPanel") -> None:
        super().__init__()
        self.panel = panel
        self.setObjectName("inspectTab")
        self._roi_id: str | None = None
        self._boxes: list[dict | None] = []
        self.form = qt.QFormLayout(self)

        self.task = qt.QComboBox()
        self.task.addItems(list(TASKS))
        self.task.setToolTip("Set from the app; change it to inspect another kind of result")
        self.form.addRow("Task", self.task)
        self.first = self._selector("InspectFirst", SCALAR)
        self.second = self._selector("InspectSecond", LABELS)
        self.transform = self._selector("InspectTransform", ["vtkMRMLTransformNode"])
        self.transform.setToolTip("The transform that brings the moving image onto the fixed one")
        self.form.addRow("Image", self.first)
        self.form.addRow("Result", self.second)
        self.form.addRow("Transform", self.transform)
        self.task.currentIndexChanged.connect(self._on_task_changed)

        settings = qt.QSettings()
        self.inspector = qt.QComboBox()
        for name, _provider, _cloud in INSPECTORS:
            self.inspector.addItem(name)
        self.inspector.setCurrentIndex(min(int(settings.value(SETTINGS + "Inspector", 0)), len(INSPECTORS) - 1))
        self.url = qt.QLineEdit(settings.value(SETTINGS + "Url", "http://localhost:8000/v1"))
        self.url.setToolTip("An OpenAI-compatible server (vLLM, Ollama) on this machine or your network")
        self.model = qt.QLineEdit(settings.value(SETTINGS + "Model", ""))
        self.model.setPlaceholderText("model name")
        row = qt.QHBoxLayout()
        for widget in (self.inspector, self.url, self.model):
            row.addWidget(widget)
        self.form.addRow("Inspector", row)
        self.inspector.currentIndexChanged.connect(self._on_inspector_changed)
        self._on_inspector_changed()
        self.depth_choice = qt.QComboBox()
        for _depth, text in DEPTHS:
            self.depth_choice.addItem(text)
        self.depth_choice.setCurrentIndex(min(int(settings.value(SETTINGS + "Depth", 1)), len(DEPTHS) - 1))
        self.depth_choice.setToolTip("Deeper finds more and takes longer (and costs more with a cloud model)")
        self.form.addRow("Depth", self.depth_choice)

        self.verdict = qt.QLabel("")
        self.verdict.setWordWrap(True)
        self.form.addRow(self.verdict)
        self.findings = qt.QTableWidget(0, 3)
        self.findings.setHorizontalHeaderLabels(["Severity", "Defect", "What the inspector saw"])
        self.findings.horizontalHeader().setStretchLastSection(True)
        self.findings.verticalHeader().setVisible(False)
        self.findings.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.findings.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.findings.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.findings.setToolTip("Click a finding: the slice views go to it")
        self.findings.currentCellChanged.connect(lambda row, *_: self._go_to(row))
        self.form.addRow(self.findings)
        self._on_task_changed()

    def _selector(self, key: str, node_types: list[str]):
        """A node selector kept in the parameter node: its change updates the run button as every selector's does."""
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = node_types
        selector.addEnabled = selector.removeEnabled = selector.renameEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        selector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            lambda node: self.panel.set_parameter_node(key, node.GetID() if node is not None else None),
        )
        return selector

    def _on_task_changed(self) -> None:
        (_, first), (_, second), types = TASKS[self.task.currentText]
        self.form.labelForField(self.first).setText(first)
        self.form.labelForField(self.second).setText(second)
        if list(self.second.nodeTypes) != types:
            self.second.nodeTypes = types
        registration = self.task.currentText == "registration"
        self.transform.setVisible(registration)
        self.form.labelForField(self.transform).setVisible(registration)
        self.panel.set_parameter("InspectTask", self.task.currentText)  # the run button follows, as on any change

    def _on_inspector_changed(self) -> None:
        local_server = INSPECTORS[self.inspector.currentIndex][1] == "openai"
        self.url.setVisible(local_server)
        self.model.setVisible(local_server)

    def on_app_changed(self, app) -> None:
        task = (getattr(app, "get_task", lambda: None)() or "").lower()
        if task in TASKS:
            self.task.setCurrentText(task)

    def prefill(self) -> None:
        """What the app has just run on, unless the user chose otherwise."""
        for selector, key in ((self.first, "InputVolume"), (self.second, "OutputVolume")):
            node = self.panel.get_parameter_node(key)
            if selector.currentNode() is None and node is not None and node.GetClassName() in selector.nodeTypes:
                selector.setCurrentNode(node)

    def inputs_ok(self) -> bool:
        needed = [self.second.currentNode()]
        if self.task.currentText == "registration":
            needed.append(self.transform.currentNode())
        return has_node_content(self.first.currentNode()) and all(node is not None for node in needed)

    def run(self, remote_server, devices: list[str]) -> None:
        """Export the case, run ``konfai-scout inspect`` on it, show its report."""
        name, provider, cloud = INSPECTORS[self.inspector.currentIndex]
        command = shutil.which("konfai-scout")
        if command is None:
            return self._stop("[KonfAI] konfai-scout is not installed in Slicer's Python (pip install konfai-scout).")
        if cloud and not slicer.util.confirmOkCancelDisplay(
            "The images of this case will be sent to Anthropic's servers to be inspected.\n\n"
            "Do not continue with data you are not allowed to share.",
            windowTitle="Images leave this machine",
        ):
            return self._stop("[KonfAI] Inspection cancelled: nothing was sent.")
        settings = qt.QSettings()
        settings.setValue(SETTINGS + "Inspector", self.inspector.currentIndex)
        settings.setValue(SETTINGS + "Url", self.url.text)
        settings.setValue(SETTINGS + "Model", self.model.text)
        settings.setValue(SETTINGS + "Depth", self.depth_choice.currentIndex)
        self._show(None)

        task = self.task.currentText
        (first_role, _), (second_role, _), _ = TASKS[task]
        work = self.panel._work_dir
        first, second = self.first.currentNode(), self.second.currentNode()
        args = ["inspect", "--task", task, "--name", second.GetName(), "--out", "Inspect", "--provider", provider]
        args += ["--depth", DEPTHS[self.depth_choice.currentIndex][0]]
        if task == "segmentation":
            labels = self._export_labels(second, first, work / "prediction.mha")
            args += ["--labels", labels] if labels else []
        else:
            self.panel._write_volume(second, work / f"{second_role}.mha")
        if task == "registration":  # Slicer writes it in ITK's convention, the fixed point to the moving one
            slicer.util.saveNode(self.transform.currentNode(), str(work / "transform.h5"))
            args += ["--input", "transform=transform.h5"]
        self.panel._write_volume(first, work / f"{first_role}.mha")
        args += ["--input", f"{first_role}={first_role}.mha", "--input", f"{second_role}={second_role}.mha"]
        if provider == "openai":
            # ponytail: a 32k window assumed (the trained model on a 24 GB card); the app should declare its own
            args += ["--base-url", self.url.text, "--local-provider", "--max-context-tokens", "26000"]
            args += ["--model", self.model.text] if self.model.text else []
        if cloud:
            args.append("--allow-cloud-images")
        self.panel._update_logs(f"[KonfAI] Inspection of a {task} by: {name}.", False)

        def on_end() -> None:
            report = work / "Inspect" / "report.json"
            if not report.is_file():
                self.panel._update_logs("[KonfAI] The inspection left no report: see the lines above.", False)
                return
            self._show(json.loads(report.read_text()))

        self.panel.process.run(command, work, args, on_end)

    def _export_labels(self, result, image, path) -> str:
        """A segmentation as a label map on the image's grid (Scout views a case on one grid); its label names."""
        labelmap = result
        if result.IsA("vtkMRMLSegmentationNode"):
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "KonfAIInspect")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(result, labelmap, image)
        colors = labelmap.GetDisplayNode().GetColorNode() if labelmap.GetDisplayNode() else None
        values = [int(v) for v in np.unique(slicer.util.arrayFromVolume(labelmap)) if v != 0]
        self.panel._write_volume(labelmap, path)
        if labelmap is not result:
            slicer.mrmlScene.RemoveNode(labelmap)
        return ",".join(
            f"{v}={(colors.GetColorName(v) if colors else str(v)).replace(',', ' ').replace('=', ' ')}" for v in values
        )

    def _stop(self, message: str) -> None:
        self.panel._update_logs(message, False)
        self.panel.set_running(False)

    def _show(self, report: dict | None) -> None:
        """The verdict and one row a finding; ``None`` empties the tab."""
        self.findings.setRowCount(0)
        self._boxes = []
        if self._roi_id and (roi := slicer.mrmlScene.GetNodeByID(self._roi_id)):  # the last report's box goes with it
            slicer.mrmlScene.RemoveNode(roi)
        self._roi_id = None
        self.verdict.setText("")
        if report is None:
            return
        if report.get("error"):
            self.verdict.setText(f"The inspection stopped: {report['error']}")
        elif conclusion := report.get("conclusion"):
            self.verdict.setText(f"<b>{conclusion['verdict'].replace('_', ' ').capitalize()}.</b> {conclusion['summary']}")
        for observation in report.get("observations", []):
            row = self.findings.rowCount
            self.findings.insertRow(row)
            kind = observation["kind"]
            cells = (
                observation["severity"] if kind != "measured_property" else "measure",
                observation["defect"].replace("_", " ") + (" (suspected)" if kind == "suspected_error" else ""),
                observation["description"],
            )
            for column, text in enumerate(cells):
                item = qt.QTableWidgetItem(text)
                item.setToolTip(observation["description"])
                self.findings.setItem(row, column, item)
            self._boxes.append(observation.get("box_mm"))
        self.findings.resizeColumnToContents(0)
        self.findings.resizeColumnToContents(1)

    def _go_to(self, row: int) -> None:
        """The slice views on the finding, its box drawn. Scout speaks LPS millimetres, the scene RAS."""
        box = self._boxes[row] if 0 <= row < len(self._boxes) else None
        if box is None:
            return
        low, high = np.array(box["min"]), np.array(box["max"])
        center = (low + high) / 2 * (-1, -1, 1)
        roi = slicer.mrmlScene.GetNodeByID(self._roi_id) if self._roi_id else None
        if roi is None:
            roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "Scout finding")
            roi.SetLocked(True)
            roi.GetDisplayNode().SetFillVisibility(False)
            self._roi_id = roi.GetID()
        roi.SetCenter(*center)
        roi.SetSize(*(high - low))
        slicer.modules.markups.logic().JumpSlicesToLocation(*center, True)
