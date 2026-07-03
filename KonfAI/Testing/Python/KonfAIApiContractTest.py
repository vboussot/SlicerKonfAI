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

"""API contract test for the KonfAI module.

SlicerImpactSynth and SlicerImpactReg import symbols from this module and
subclass some of them. This test freezes that contract so any refactoring
of KonfAI keeps the sister extensions working:

- ImpactSynth.py:
    from KonfAI import KonfAIAppTemplateWidget, KonfAICoreWidget, _is_reload_setup
- ImpactReg.py:
    from KonfAI import (AppTemplateWidget, ChipSelector, KonfAICoreWidget,
                        KonfAIMetricsPanel, Process, RemoteServer,
                        _is_reload_setup, has_node_content)

ImpactReg additionally subclasses AppTemplateWidget and Process and relies on
protected members and callback signatures (see each test below).

This test must run inside Slicer (registered with slicer_add_python_unittest).
"""

import inspect
import sys
import unittest
from pathlib import Path

# Make KonfAI.py importable when the test is run from the build tree.
_MODULE_DIR = str(Path(__file__).resolve().parents[2])
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)


class KonfAIApiContractTest(unittest.TestCase):
    def test_public_imports(self):
        """The exact import lines of ImpactSynth.py:17 and ImpactReg.py:30-39."""
        from KonfAI import (  # noqa: F401
            AppTemplateWidget,
            ChipSelector,
            KonfAIAppTemplateWidget,
            KonfAICoreWidget,
            KonfAIMetricsPanel,
            Process,
            RemoteServer,
            _is_reload_setup,
            has_node_content,
        )

    def test_process_protocol(self):
        """ElastixProcess(ImpactReg.py) subclasses Process and uses its
        __init__(update_logs, update_progress, running_setter) plus the
        protected attributes _update_logs/_update_progress."""
        from KonfAI import Process

        params = list(inspect.signature(Process.__init__).parameters)
        self.assertEqual(params[:4], ["self", "update_logs", "update_progress", "running_setter"])

        process = Process(lambda text, clear=None: None, lambda value, speed=None: None, lambda running: None)
        self.assertTrue(hasattr(process, "_update_logs"))
        self.assertTrue(hasattr(process, "_update_progress"))
        self.assertTrue(callable(process.run))
        self.assertTrue(callable(process.stop))

    def test_app_template_widget_protocol(self):
        """ElastixImpactWidget(ImpactReg.py) subclasses AppTemplateWidget:
        overrides app_setup (called with positional args by register_apps),
        uses on_run_button(function) where function(remote_server, devices),
        and touches _work_dir/_name/_parameter_node/_initialized."""
        from KonfAI import AppTemplateWidget

        params = list(inspect.signature(AppTemplateWidget.app_setup).parameters)
        self.assertEqual(
            params,
            [
                "self",
                "update_logs",
                "update_progress",
                "parameter_node",
                "begin_status_progress",
                "end_status_progress",
            ],
        )
        for method in ("on_run_button", "get_work_dir", "get_remote_server", "enter", "exit", "cleanup"):
            self.assertTrue(callable(getattr(AppTemplateWidget, method, None)), f"missing method: {method}")

    def test_remote_server_protocol(self):
        """ImpactReg builds RemoteServer(name, host, port) instances.
        Signature check only: __init__ has a keyring side effect."""
        from KonfAI import RemoteServer

        params = list(inspect.signature(RemoteServer.__init__).parameters)
        self.assertEqual(params[:4], ["self", "name", "host", "port"])
        for method in ("get_headers", "get_url"):
            self.assertTrue(callable(getattr(RemoteServer, method, None)), f"missing method: {method}")

    def test_core_widget_protocol(self):
        """Both sisters instantiate KonfAICoreWidget(title) and call
        register_apps(list_of_apps); register_apps drives app.app_setup."""
        from KonfAI import KonfAICoreWidget

        self.assertTrue(callable(getattr(KonfAICoreWidget, "register_apps", None)))
        for method in ("enter", "exit", "cleanup"):
            self.assertTrue(callable(getattr(KonfAICoreWidget, method, None)), f"missing method: {method}")

    def test_helpers(self):
        from KonfAI import _is_reload_setup, has_node_content

        self.assertTrue(callable(_is_reload_setup))
        self.assertTrue(callable(has_node_content))


if __name__ == "__main__":
    unittest.main()
