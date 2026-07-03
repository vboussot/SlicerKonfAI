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

"""Small GUI helpers shared across KonfAI widgets."""

import os
from contextlib import contextmanager

import slicer
from qt import QMessageBox


def ask_user_to_install_dependency(package_label: str, details: str) -> bool:
    """
    Ask the user for permission to install a dependency.
    Returns True if user accepts, False otherwise.
    """
    mb = QMessageBox(slicer.util.mainWindow())
    mb.setIcon(QMessageBox.Question)
    mb.setWindowTitle("Additional dependency required")
    mb.setText(f"This module requires {package_label}.")
    mb.setInformativeText(details + "\n\nIf you choose 'No', the module will close.")
    mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    mb.setDefaultButton(QMessageBox.Yes)
    return mb.exec_() == QMessageBox.Yes


@contextmanager
def slicer_wait_popup(title: str, text: str):
    """
    Show a modal 'please wait' dialog during long blocking operations.

    This prevents users from force-quitting Slicer because the UI looks frozen.
    """
    box = QMessageBox(slicer.util.mainWindow())
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QMessageBox.Information)
    box.setStandardButtons(QMessageBox.NoButton)  # no buttons = pure wait popup
    box.setModal(True)
    box.show()

    # Let Qt process paint events so the dialog is actually displayed
    slicer.app.processEvents()

    try:
        yield
    finally:
        box.hide()
        box.deleteLater()
        slicer.app.processEvents()


def resource_path(filename: str) -> str:
    """
    Return the absolute path of a file located in the module's `Resources` directory.

    Parameters
    ----------
    filename : str
        File name relative to the `Resources` directory.

    Returns
    -------
    str
        Absolute path to the requested resource file.
    """
    scripted_modules_path = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scripted_modules_path, "Resources", filename)
