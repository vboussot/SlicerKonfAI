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

"""Base class for KonfAI application sub-panels.

A panel is a child widget of ``KonfAIAppTemplateWidget``: it loads its own
.ui file, connects its own signals, and delegates all shared state
(parameter node, process, work directory, logging) to the owning template.
"""

from typing import TYPE_CHECKING

import slicer
from qt import QVBoxLayout, QWidget

from KonfAILib.widgets.helpers import resource_path

if TYPE_CHECKING:
    from KonfAILib.widgets.app_template import KonfAIAppTemplateWidget


class KonfAIAppPanel(QWidget):
    """Child panel of KonfAIAppTemplateWidget. Loads its .ui, delegates state to the template."""

    def __init__(self, template: "KonfAIAppTemplateWidget", ui_name: str) -> None:
        super().__init__()
        self.template = template
        ui_widget = slicer.util.loadUI(resource_path(f"UI/{ui_name}.ui"))
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        # Bind the MRML scene so that qMRML widgets in the .ui file work properly
        if hasattr(ui_widget, "setMRMLScene"):
            ui_widget.setMRMLScene(slicer.mrmlScene)

    # ---- Delegations: method bodies moved from the template compile unchanged ----
    def set_parameter(self, key, value):
        self.template.set_parameter(key, value)

    def get_parameter(self, key):
        return self.template.get_parameter(key)

    def set_parameter_node(self, key, value):
        self.template.set_parameter_node(key, value)

    def get_parameter_node(self, key):
        return self.template.get_parameter_node(key)

    def get_remote_server(self):
        return self.template.get_remote_server()

    def is_running(self):
        return self.template.is_running()

    def set_running(self, state):
        self.template.set_running(state)

    def on_run_button(self, fn):
        # fn = bound method of the panel
        self.template.on_run_button(fn)

    def transient_status_progress(self, msg):
        return self.template.transient_status_progress(msg)

    def _set_status_progress(self, v, m):
        self.template._set_status_progress(v, m)

    @property
    def process(self):
        return self.template.process

    @property
    def _work_dir(self):
        return self.template._work_dir

    @property
    def _name(self):
        return self.template._name

    @property
    def _update_logs(self):
        return self.template._update_logs

    @property
    def _update_progress(self):
        return self.template._update_progress

    # ---- Downward protocol (no-op by default) ----
    def on_app_changed(self, app) -> None:
        pass

    def initialize_gui_from_parameter_node(self) -> None:
        pass

    def update_gui_from_parameter_node(self) -> None:
        pass

    def update_parameter_node_from_gui(self) -> None:
        pass

    def enter(self) -> None:
        pass

    def exit(self) -> None:  # noqa: A003
        pass

    def cleanup(self) -> None:
        pass
