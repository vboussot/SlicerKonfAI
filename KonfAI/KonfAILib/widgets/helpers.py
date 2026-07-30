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
from qt import QIcon, QMessageBox, QPalette


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


def app_task_icon(app) -> QIcon:
    """Icon for an app entry in the selector: the bundle's own icon when it ships one (``icon.png``
    or the ``icon`` file named in app.json), otherwise a glyph for its declared task family.

    Unknown or undeclared tasks fall back to the neutral app icon, so the selector renders
    consistently whatever the manifest says.
    """
    icon_path = getattr(app, "get_icon_path", lambda: None)()
    if icon_path:
        return QIcon(str(icon_path))
    task = (getattr(app, "get_task", lambda: None)() or "").lower()
    name = task if task in ("segmentation", "registration", "synthesis") else "app"
    return QIcon(resource_path(f"Icons/task_{name}.png"))


# Shared canvas-and-cards style for KonfAI dialogs: cards on a muted canvas instead of flat forms.
# Slicer themes are palette-based, so each dialog picks its light or dark color set from the palette.
# Scroll areas must stay transparent, otherwise their viewport paints over the canvas the cards sit on.
_DIALOG_QSS_TEMPLATE = """
QDialog {{ background: {canvas}; }}
QScrollArea {{ background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QFrame#card {{ background: {card}; border: 1px solid {border}; border-radius: 8px; }}
QLabel#cardHeader {{ font-weight: 600; color: {heading}; }}
QLabel#sectionHeader {{ font-weight: 600; color: {heading}; }}
QLabel#dialogSection {{ color: {muted}; font-weight: 700; font-size: 11px; letter-spacing: 1px; margin-top: 4px; }}
QLabel#planHint {{ color: {muted}; font-style: italic; }}
QLabel#dim {{ color: {muted}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    border: 1px solid {field_border}; border-radius: 5px; padding: 3px 6px; background: {field}; min-height: 22px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border: 1px solid #3b82f6; }}
QListWidget {{ border: 1px solid {field_border}; border-radius: 6px; background: {field}; padding: 4px; }}
QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
QListWidget::item:hover {{ background: {hover}; }}
QListWidget::item:selected {{ background: #3b82f6; color: white; }}
QPushButton#ghost {{
    border: 1px solid {field_border}; border-radius: 6px; padding: 4px 12px; background: {card}; color: {heading};
}}
QPushButton#ghost:hover {{ background: {hover}; }}
QPushButton#remove {{ border: none; color: {remove}; font-weight: 600; padding: 2px 8px; background: transparent; }}
QPushButton#remove:hover {{ color: {remove_hover}; }}
"""

_DIALOG_COLORS_LIGHT = dict(
    canvas="#f4f5f7", card="#ffffff", border="#dfe3e8", heading="#2b3440", muted="#8a94a3",
    field="#ffffff", field_border="#cfd5dc", hover="#eef2f6", remove="#b64a4a", remove_hover="#d33333",
)
_DIALOG_COLORS_DARK = dict(
    canvas="#2b2d30", card="#36393c", border="#46494d", heading="#e4e6e8", muted="#9aa3ad",
    field="#26282a", field_border="#55595e", hover="#41454a", remove="#e07a76", remove_hover="#f28b87",
)


def is_dark_theme() -> bool:
    """True when Slicer's palette is dark (the themes are palette-based)."""
    return slicer.app.palette().color(QPalette.Window).lightness() < 128


def themed_dialog_qss() -> str:
    """The shared dialog stylesheet, in the light or dark variant matching Slicer's palette."""
    return _DIALOG_QSS_TEMPLATE.format(**(_DIALOG_COLORS_DARK if is_dark_theme() else _DIALOG_COLORS_LIGHT))


# Primary-action look for the Run/Stop buttons. The colors are theme-neutral (accent + rgba grays)
# so they read correctly in both the light and dark Slicer themes. The red 'Stop' state is driven by
# the dynamic 'running' property, which update_gui_from_parameter_node sets and repolishes.
_RUN_BUTTON_QSS = """
QPushButton {
    background-color: #3b82f6; color: white; font-weight: 600;
    border: none; border-radius: 5px; padding: 4px 12px;
}
QPushButton:hover { background-color: #2f6fe0; }
QPushButton:pressed { background-color: #2a63c8; }
QPushButton:disabled { background-color: rgba(128,128,128,0.22); color: rgba(128,128,128,0.9); }
QPushButton[running="true"] { background-color: #d9534f; }
QPushButton[running="true"]:hover { background-color: #c9302c; }
"""


def style_run_button(button) -> None:
    """Give a Run/Stop button the primary-action style (accent blue, red while running)."""
    button.setStyleSheet(_RUN_BUTTON_QSS)


def set_run_button_running(button, running: bool) -> None:
    """Switch a styled Run/Stop button between its blue (idle) and red (running) states."""
    button.setProperty("running", "true" if running else "false")
    button.style().unpolish(button)
    button.style().polish(button)


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
