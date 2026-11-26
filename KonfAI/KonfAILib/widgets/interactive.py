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

"""Interactive segmentation with a prompted app: clicks and boxes placed in the views, a prediction after each.

A prompted app declares a ``FIDUCIALS`` input (the prompts: points labelled ``positive``, ``negative`` or ``box``, in
the order they were given) and an optional ``SEGMENTATION`` one (the last mask). Each placed, moved or deleted point
runs the app again on the image, every prompt so far and the object's current segment, which may have been edited by
hand in between; the result replaces that segment. The app runs in a session that keeps its model loaded
(``python -m konfai_apps.interactive``), in Slicer's Python or on the selected konfai-apps server, so a correction
costs a prediction, not a model load.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
from qt import QHBoxLayout, QLabel, QProcess, QProcessEnvironment, QPushButton, QVBoxLayout, QWidget

PREFIX = "konfai-interactive:"
# kind: (button, colour); the labels are the ones a prompted app reads (konfai.network.prompted)
KINDS = {
    "positive": ("+ Point", (0.18, 0.75, 0.25)),
    "negative": ("− Point", (0.9, 0.2, 0.2)),
    "box": ("Box (2 corners)", (1.0, 0.75, 0.0)),
}


def prompted_inputs(app) -> tuple[str, str | None] | None:
    """The keys of a prompted app's prompts input and last-mask input (None if it takes none), or None for an app
    that takes no prompts."""
    types = {
        key: str(getattr(entry.volume_type, "value", entry.volume_type)) for key, entry in app.get_inputs().items()
    }
    prompts = next((key for key, kind in types.items() if kind == "FIDUCIALS"), None)
    if prompts is None:
        return None
    return prompts, next((key for key, kind in types.items() if kind == "SEGMENTATION"), None)


class InteractiveSegmentation(QWidget):
    def __init__(self, panel) -> None:
        super().__init__()
        self.panel = panel
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.buttons = {}
        for kind, (text, _) in KINDS.items():
            button = QPushButton(text)
            button.checkable = True
            button.toolTip = f"Place {kind} prompts in the views (click again to stop)"
            button.toggled.connect(lambda checked, k=kind: self._on_kind(k, checked))
            row.addWidget(button)
            self.buttons[kind] = button
        self.layout().addLayout(row)
        row = QHBoxLayout()
        new_object = QPushButton("Keep and new object")
        new_object.toolTip = "Keep this object's segment and start the next one"
        new_object.clicked.connect(self.new_object)
        reset = QPushButton("Reset")
        reset.toolTip = "Delete this object's prompts and empty its segment"
        reset.clicked.connect(self.reset)
        row.addWidget(new_object)
        row.addWidget(reset)
        self.layout().addLayout(row)
        self.status = QLabel("Place a point to segment.")
        self.layout().addWidget(self.status)

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)  # an unread stderr would fill and stall it
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_worker_finished)
        self._worker_key = None  # (app, devices) the running worker serves
        self._ready = False
        self._busy = False  # a request sent, its answer not yet read
        self._pending = False  # prompts changed while a request ran
        self._request = 0
        self._app = None
        self._keys: tuple[str, str | None] | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="konfai-interactive-"))
        self._nodes: dict[str, object] = {}
        self._observers: list[tuple[object, int]] = []
        self._order: list[tuple[str, str]] = []  # (kind, control point ID), in the order they were given
        self._volume = None  # (node ID, image MTime) of the exported volume, and its SimpleITK header
        self._header = None
        self._segment = None  # the current object's segment ID
        self._sent = None  # what the last request was made of: one click fires several events
        self._target: tuple = ()  # the volume, segmentation and segment the request in progress answers into

    # ---- app and worker ---------------------------------------------------------------------------------------

    def set_app(self, app) -> bool:
        """Follow the selected app: shown, and its worker started at the first prompt, only for a prompted one."""
        keys = prompted_inputs(app) if app is not None else None
        if app is not self._app:
            self._stop_worker()
            self._app, self._keys = app, keys
        self.setVisible(keys is not None)
        if keys is None:
            self._place(None)
        return keys is not None

    def _start_worker(self) -> None:
        devices = self.panel.template.get_device()
        server, _ = self.panel.template.get_remote_server()
        key = (self._app.get_name(), tuple(devices), str(server))
        if self._worker_key == key and self.process.state() != QProcess.NotRunning:
            return
        self._stop_worker()
        self._worker_key, self._ready = key, False
        args = ["-m", "konfai_apps.interactive", self._app.get_name()]
        if devices:
            args += ["--gpu", *devices]
        environment = QProcessEnvironment.systemEnvironment()
        if server is not None:
            options = server.cli_args()
            if server.token:  # by the environment: a command line is readable by every user of the machine
                environment.insert("KONFAI_API_TOKEN", server.token)
                del options[options.index("--token") : options.index("--token") + 2]
            args += options
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self._dir))
        self.process.start(sys.executable, args)
        where = f" on {server.name}" if server is not None else ""
        self.status.setText(f"Loading {self._app.get_name()}{where}…")

    def _stop_worker(self) -> None:
        self._worker_key = None  # a stop of ours, not a crash
        if self.process.state() != QProcess.NotRunning:
            # Its input closed, the worker ends: a server's session is closed with it, its model's memory freed.
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(5000):
                self.process.kill()
                self.process.waitForFinished(2000)
        self._ready, self._busy, self._pending = False, False, False

    def _on_worker_finished(self, *_) -> None:
        if self._worker_key is not None:  # not stopped by us
            self.status.setText("The app's worker stopped; see the Python console. Place a point to restart it.")
        self._worker_key, self._ready, self._busy, self._sent = None, False, False, None

    def _on_output(self) -> None:
        while self.process.canReadLine():
            line = self.process.readLine().data().decode(errors="replace").rstrip()
            if not line.startswith(PREFIX):
                print(line)
                continue
            word, _, rest = line[len(PREFIX) :].strip().partition(" ")
            if word == "ready":
                self._ready = True
                self._run()
                continue
            request, _, detail = rest.partition(" ")
            self._busy = False
            output = self._dir / f"out{request}"
            if int(request) != self._request:
                pass  # an answer for prompts reset since: nothing to show
            elif word == "done":
                self._load(output)
                self.status.setText(f"Segmented in {float(detail):.1f} s ({len(self._order)} prompts).")
            else:
                self._sent = None  # the same prompts may be tried again
                self.status.setText(f"The app refused: {detail}")
            shutil.rmtree(output, ignore_errors=True)  # loaded, or no longer wanted
            if self._pending:
                self._pending = False
                self._run()

    # ---- prompts ----------------------------------------------------------------------------------------------

    def _node(self, kind: str):
        node = self._nodes.get(kind)
        if node is None or slicer.mrmlScene.GetNodeByID(node.GetID()) is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"KonfAI {kind}")
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSelectedColor(*KINDS[kind][1])
            node.GetDisplayNode().SetColor(*KINDS[kind][1])
            node.GetDisplayNode().SetPointLabelsVisibility(False)  # the colour tells the kind
            node.SetAttribute("KonfAI.kind", kind)
            for event, callback in (
                (slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._on_point_added),
                (slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, lambda *_: self.changed()),
                (slicer.vtkMRMLMarkupsNode.PointRemovedEvent, lambda *_: self.changed()),
            ):
                self._observers.append((node, node.AddObserver(event, callback)))
            self._nodes[kind] = node
        return node

    def _uncheck(self, keep: str | None = None) -> None:
        for kind, button in self.buttons.items():
            if kind != keep:
                button.blockSignals(True)
                button.checked = False
                button.blockSignals(False)

    def _on_kind(self, kind: str, checked: bool) -> None:
        if checked:
            self._uncheck(keep=kind)
        self._place(kind if checked else None)

    def _place(self, kind: str | None) -> None:
        """Put the views in persistent place mode on ``kind``'s points, or back in view mode."""
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        if kind is None:
            self._uncheck()
            interaction.SetCurrentInteractionMode(interaction.ViewTransform)
            return
        node = self._node(kind)
        slicer.app.applicationLogic().GetSelectionNode().SetReferenceActivePlaceNodeClassName(node.GetClassName())
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(node.GetID())
        interaction.SetPlaceModePersistence(1)
        interaction.SetCurrentInteractionMode(interaction.Place)

    def _on_point_added(self, node, _event) -> None:
        self._order.append(
            (node.GetAttribute("KonfAI.kind"), node.GetNthControlPointID(node.GetNumberOfControlPoints() - 1))
        )
        self.changed()

    def _prompts(self) -> tuple[np.ndarray, list[str]]:
        """Every prompt still in the scene, in LPS, with its kind, in the order given; a point deleted drops out."""
        points, kinds, kept = [], [], []
        for kind, point_id in self._order:
            node = self._nodes.get(kind)
            index = node.GetNthControlPointIndexByID(point_id) if node is not None else -1
            if index < 0:
                continue
            ras = node.GetNthControlPointPositionWorld(index)
            points.append([-ras[0], -ras[1], ras[2]])
            kinds.append(kind)
            kept.append((kind, point_id))
        self._order = kept
        return np.array(points, dtype=float).reshape(-1, 3), kinds

    # ---- runs -------------------------------------------------------------------------------------------------

    def changed(self) -> None:
        """The prompts changed: run again, now or once the run in progress answers."""
        if self._keys is None:
            return
        if self._busy:
            self._pending = True
            return
        self._run()

    def _run(self) -> None:
        points, kinds = self._prompts()
        if not len(points) or kinds.count("box") % 2:  # nothing to show yet, or a box waiting for its second corner
            return
        volume_node = self.panel.ui.inputVolumeSelector.currentNode()
        if volume_node is None:
            self.status.setText("Select the image to segment.")
            return
        self._start_worker()
        if not self._ready:
            return  # sent once the worker says it is ready
        from konfai.utils.dataset import write_landmarks

        state = (volume_node.GetID(), volume_node.GetImageData().GetMTime())
        sent = (points.tobytes(), tuple(kinds), state)
        if sent == self._sent:
            return
        if self._volume != state:
            image = sitkUtils.PullVolumeFromSlicer(volume_node)
            sitk.WriteImage(image, str(self._dir / "Volume.mha"))
            self._volume, self._header = state, image
        prompts_key, previous_key = self._keys
        write_landmarks(points, self._dir / f"{prompts_key}.fcsv", kinds)
        inputs = [[str(self._dir / "Volume.mha")], [str(self._dir / f"{prompts_key}.fcsv")]]
        segmentation, segment = self._current_segment(volume_node)
        if previous_key is not None:
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segment, volume_node)
            if mask is not None and mask.any():
                previous = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
                previous.CopyInformation(self._header)
                sitk.WriteImage(previous, str(self._dir / f"{previous_key}.mha"))
                inputs.append([str(self._dir / f"{previous_key}.mha")])
        self._request += 1
        output = self._dir / f"out{self._request}"
        self._busy, self._sent, self._target = True, sent, (volume_node, segmentation, segment)
        self.status.setText("Segmenting…")
        request = {"id": self._request, "inputs": inputs, "output": str(output)}
        self.process.write((json.dumps(request) + "\n").encode())

    def _current_segment(self, volume_node):
        """The output segmentation and the current object's segment in it, made if missing."""
        segmentation = self.panel.ui.outputVolumeSelector.currentNode()
        if segmentation is None or not segmentation.IsA("vtkMRMLSegmentationNode"):
            segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{self._app.get_name()}")
            segmentation.CreateDefaultDisplayNodes()
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            self.panel.ui.outputVolumeSelector.setCurrentNode(segmentation)
        if self._segment is None or segmentation.GetSegmentation().GetSegment(self._segment) is None:
            count = segmentation.GetSegmentation().GetNumberOfSegments()
            self._segment = segmentation.GetSegmentation().AddEmptySegment("", f"Object {count + 1}")
        return segmentation, self._segment

    def _load(self, output: Path) -> None:
        found = sorted(output.rglob("*.mha"))
        if not found:
            self.status.setText(f"The app wrote no image under {output}.")
            return
        # Into what the request was made for, whatever the selectors show now.
        volume_node, segmentation, segment = self._target
        if not all(slicer.mrmlScene.IsNodePresent(node) for node in (volume_node, segmentation)):
            self.status.setText("The image or the segmentation of the request is gone.")
            return
        if segmentation.GetSegmentation().GetSegment(segment) is None:
            self.status.setText("The segment of the request is gone.")
            return
        mask = (sitk.GetArrayFromImage(sitk.ReadImage(str(found[0]))) > 0).astype(np.uint8)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, segment, volume_node)

    # ---- objects ----------------------------------------------------------------------------------------------

    def _clear_prompts(self) -> None:
        for node in self._nodes.values():
            if slicer.mrmlScene.GetNodeByID(node.GetID()) is not None:
                node.RemoveAllControlPoints()
        self._order = []
        self._pending, self._sent = False, None
        self._request += 1  # the answer of a run in progress is for prompts that are gone

    def reset(self) -> None:
        self._clear_prompts()
        segmentation = self.panel.ui.outputVolumeSelector.currentNode()
        if self._segment is not None and segmentation is not None and segmentation.IsA("vtkMRMLSegmentationNode"):
            volume_node = self.panel.ui.inputVolumeSelector.currentNode()
            if volume_node is not None and segmentation.GetSegmentation().GetSegment(self._segment) is not None:
                empty = np.zeros(slicer.util.arrayFromVolume(volume_node).shape, np.uint8)
                slicer.util.updateSegmentBinaryLabelmapFromArray(empty, segmentation, self._segment, volume_node)
        self.status.setText("Place a point to segment.")

    def new_object(self) -> None:
        self._clear_prompts()
        self._segment = None
        self.status.setText("Next object: place a point.")

    def shutdown(self) -> None:
        self._place(None)
        self._stop_worker()
        for node, tag in self._observers:
            node.RemoveObserver(tag)
        self._observers = []
        shutil.rmtree(self._dir, ignore_errors=True)
