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

"""Dialogs to add and edit remote computation servers."""

import slicer
from qt import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from KonfAILib.logic.servers import SERVICE, RemoteServer


class RemoteServerConfigDialog(QDialog):

    def __init__(self, remote_server: RemoteServer):
        super().__init__()
        self.remote_server = remote_server
        self.setWindowTitle(f"Configure server: {remote_server.name}")
        self.setModal(True)

        self._remove = False  # <-- flag

        self.hostEdit = QLineEdit(remote_server.host)
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(int(remote_server.port))
        self.tokenEdit = QLineEdit(remote_server.token)
        self.tokenEdit.setPlaceholderText("Optional (Bearer token)")
        self.tokenEdit.setEchoMode(QLineEdit.EchoMode.Password)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)

        self.checkButton = QPushButton("Check")
        self.saveButton = QPushButton("Save")
        self.removeButton = QPushButton("Remove")
        self.cancelButton = QPushButton("Cancel")

        form = QFormLayout()
        form.addRow("Host", self.hostEdit)
        form.addRow("Port", self.portSpin)
        form.addRow("Token", self.tokenEdit)

        btns = QHBoxLayout()
        btns.addWidget(self.checkButton)
        btns.addStretch(1)
        btns.addWidget(self.removeButton)
        btns.addWidget(self.saveButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.statusLabel)
        root.addLayout(btns)

        self.cancelButton.clicked.connect(lambda _=False: self.reject())
        self.saveButton.clicked.connect(lambda _=False: self.on_save())
        self.removeButton.clicked.connect(lambda _=False: self.on_remove())
        self.checkButton.clicked.connect(lambda _=False: self.on_check())

    def on_check(self) -> bool:
        self.remote_server.host = self.hostEdit.text.strip()
        self.remote_server.port = int(self.portSpin.value)
        self.remote_server.token = self.tokenEdit.text.strip()
        from konfai import check_server

        ok, msg = check_server(self.remote_server)
        self.statusLabel.setText(("✅ " if ok else "❌ ") + msg)
        return ok

    def on_remove(self):
        self._remove = True
        self.accept()

    def on_save(self):
        if self.on_check():
            self.accept()

    def get(self) -> RemoteServer:
        self.remote_server.host = self.hostEdit.text.strip()
        self.remote_server.port = int(self.portSpin.value)
        self.remote_server.token = self.tokenEdit.text.strip()
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring  # noqa: F811

        keyring.set_password(SERVICE, str(self.remote_server), self.remote_server.token)
        return self.remote_server

    def want_remove(self) -> bool:
        return self._remove


class RemoteServerAddDialog(QDialog):
    def __init__(self, remote_servers_name: list[str]):
        super().__init__()
        self.remote_servers_name = remote_servers_name
        self.setWindowTitle("Add remote server")
        self.setModal(True)

        self.nameEdit = QLineEdit("")
        self.hostEdit = QLineEdit("127.0.0.1")
        self.portSpin = QSpinBox()
        self.portSpin.setRange(1, 65535)
        self.portSpin.setValue(8000)
        self.tokenEdit = QLineEdit("")
        self.tokenEdit.setPlaceholderText("Optional (Bearer token)")
        self.tokenEdit.setEchoMode(QLineEdit.EchoMode.Password)

        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)

        self.checkButton = QPushButton("Check")
        self.addButton = QPushButton("Add")
        self.cancelButton = QPushButton("Cancel")

        form = QFormLayout()
        form.addRow("Name", self.nameEdit)
        form.addRow("Host", self.hostEdit)
        form.addRow("Port", self.portSpin)
        form.addRow("Token", self.tokenEdit)

        btns = QHBoxLayout()
        btns.addWidget(self.checkButton)
        btns.addStretch(1)
        btns.addWidget(self.addButton)
        btns.addWidget(self.cancelButton)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.statusLabel)
        root.addLayout(btns)

        self.checkButton.clicked.connect(self.on_check)
        self.addButton.clicked.connect(self.on_add)
        self.cancelButton.clicked.connect(lambda _=False: self.reject())

    def on_check(self) -> bool:
        remote_server = self.get()
        from konfai import check_server

        ok, msg = check_server(remote_server)
        self.statusLabel.setText(("✅ " if ok else "❌ ") + msg)
        return ok

    def on_add(self):
        name = self.nameEdit.text.strip()
        if not name:
            self.statusLabel.setText("❌ Name is required")
            return
        if name in self.remote_servers_name:
            self.statusLabel.setText("❌ This server name already exists.")
            return
        if self.on_check():
            self.accept()

    def get(self) -> RemoteServer:
        name = self.nameEdit.text.strip()
        host = self.hostEdit.text.strip()
        port = int(self.portSpin.value)

        token = self.tokenEdit.text.strip() or None
        server_key = f"{name}|{host}|{port}"
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring  # noqa: F811

        if token:
            keyring.set_password(SERVICE, server_key, token)
        else:
            try:
                keyring.delete_password(SERVICE, server_key)
            except keyring.errors.PasswordDeleteError:
                pass

        return RemoteServer(name, host, port)
