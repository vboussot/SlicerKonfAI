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

"""Dialog selecting which app files from Hugging Face to (re)download."""

from collections.abc import Callable
from pathlib import Path

from qt import (
    QColor,
    QDialog,
    QHBoxLayout,
    QIcon,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSize,
    Qt,
    QVBoxLayout,
)

from KonfAILib.widgets.helpers import resource_path


class DownloadFilesDialog(QDialog):
    def __init__(
        self,
        files: list[str],
        checkpoints_name_available: list[str],
        on_open_folder: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Download from Hugging Face")
        self.setModal(True)
        self.resize(600, 450)

        self.label = QLabel("Select files to refresh:")
        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.MultiSelection)

        for f in files:
            item = QListWidgetItem(f)

            if Path(f).suffix == ".pt" and Path(f).name not in checkpoints_name_available:
                item.setForeground(QColor("#9ca3af"))
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setData(Qt.UserRole, "shadow")

            self.listw.addItem(item)

        # boutons (même style que ton exemple)
        self.openFolderButton = QPushButton()
        self.openFolderButton.setIcon(QIcon(resource_path("Icons/folder.png")))
        self.openFolderButton.setIconSize(QSize(18, 18))
        self.openFolderButton.setToolTip("Open local app folder")
        self.openFolderButton.setEnabled(on_open_folder is not None)
        self.downloadButton = QPushButton("Download")
        self.cancelButton = QPushButton("Cancel")

        btns = QHBoxLayout()
        btns.addWidget(self.openFolderButton)
        btns.addStretch(1)
        btns.addWidget(self.downloadButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addWidget(self.label)
        root.addWidget(self.listw)
        root.addLayout(btns)

        if on_open_folder is not None:
            self.openFolderButton.clicked.connect(on_open_folder)
        self.downloadButton.clicked.connect(lambda _=False: self.accept())
        self.cancelButton.clicked.connect(lambda _=False: self.reject())

        if on_open_folder is None:
            self.openFolderButton.hide()

    def selected_files(self) -> list[str]:
        return [i.text() for i in self.listw.selectedItems()]
