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

"""QA metrics panel: browse metric files and load result images."""

from pathlib import Path

import slicer
from qt import QFont, QListWidgetItem, Qt, QVBoxLayout, QWidget

from KonfAILib.widgets.helpers import resource_path


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
        from konfai.utils.dataset import get_infos

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
