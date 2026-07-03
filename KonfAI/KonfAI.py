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

# flake8: noqa: E402
import sys

import slicer
from qt import (
    QWidget,
)
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

# Slicer's developer Reload only re-executes this file: cached KonfAILib
# submodules must be dropped so they are re-imported fresh alongside it.
# After a Reload of KonfAI, ImpactSynth/ImpactReg must be reloaded too
# (in that order) to rebind to the new class identities.
for _cached_module in [name for name in sys.modules if name == "KonfAILib" or name.startswith("KonfAILib.")]:
    del sys.modules[_cached_module]

# Facade: public API of the KonfAI module. Sister extensions (ImpactSynth,
# ImpactReg) import these symbols from `KonfAI`; keep re-exports stable
# (see Testing/Python/KonfAIApiContractTest.py).
from KonfAILib import KONFAI_SLICER_API_VERSION  # noqa: F401
from KonfAILib.logic.dependencies import get_installed_version, get_latest_pypi_version  # noqa: F401
from KonfAILib.logic.dependencies import install_konfai, install_package, install_torch  # noqa: F401
from KonfAILib.logic.process import Process  # noqa: F401
from KonfAILib.logic.servers import SERVICE  # noqa: F401
from KonfAILib.logic.servers import RemoteServer  # noqa: F401
from KonfAILib.platform_utils import open_path_in_file_browser  # noqa: F401
from KonfAILib.widgets.chip_selector import ChipSelector  # noqa: F401
from KonfAILib.widgets.dialogs.download_files import DownloadFilesDialog  # noqa: F401
from KonfAILib.widgets.dialogs.remote_server import RemoteServerAddDialog, RemoteServerConfigDialog  # noqa: F401
from KonfAILib.widgets.helpers import ask_user_to_install_dependency, slicer_wait_popup  # noqa: F401
from KonfAILib.widgets.helpers import resource_path  # noqa: F401
from KonfAILib.widgets.metrics_panel import KonfAIMetricsPanel  # noqa: F401
from KonfAILib.logic.hf import get_hf_app_file_list, split_hf_repo_reference  # noqa: F401
from KonfAILib.logic.mrml import has_node_content  # noqa: F401
from KonfAILib.widgets.app_template import AppTemplateWidget, KonfAIAppTemplateWidget  # noqa: F401
from KonfAILib.widgets.core_widget import KonfAICoreWidget  # noqa: F401
from KonfAILib.widgets.panels.inference import KonfAIAppInferencePanel  # noqa: F401
from KonfAILib.widgets.panels.qa import KonfAIAppQAPanel  # noqa: F401
from KonfAILib.widgets.panels.selection import KonfAIAppSelectionPanel  # noqa: F401


class KonfAI(ScriptedLoadableModule):
    """
    Main Slicer module class for KonfAI.

    This class is responsible for registering the module in 3D Slicer,
    providing metadata (name, category, help text, acknowledgements, etc.),
    and making the module discoverable from the Slicer module list.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KonfAI")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Pipelines")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Valentin Boussot (University of Rennes, France)",
        ]
        # Short help text shown in the module panel
        self.parent.helpText = _(
            "<p>"
            "SlicerKonfAI is a 3D Slicer module that lets you run KonfAI Apps directly inside Slicer."
            "</p>"
            "<p>"
            "You can:<br>"
            "&bull; Discover and download KonfAI Apps hosted on Hugging Face<br>"
            "&bull; Run fast GPU-accelerated inference on volumes already loaded in Slicer<br>"
            "&bull; Automatically export/import data (DICOM, NRRD, NIfTI, etc.) as volumes or segmentations<br>"
            "&bull; Perform Quality Assurance (QA) using reference-based metrics or reference-free uncertainty "
            "estimation (TTA, MC dropout, multi-model ensembling)"
            "</p>"
            "<p>"
            "Each KonfAI App is a self-contained workflow "
            "(Prediction/Evaluation/Uncertainty YAML configs + trained model) "
            "that can be executed identically from Python, the CLI, or this Slicer module."
            "</p>"
            "<p><b>In short:</b><br>"
            "SlicerKonfAI = GUI + data exchange + process manager<br>"
            "KonfAI      = computation engine for training, inference, and evaluation."
            "</p>"
        )

        # Acknowledgment text (displayed in the About section)
        self.parent.acknowledgementText = _(
            "<p>This module was originally developed by Valentin Boussot "
            "(University of Rennes, France).<br>"
            "It integrates the KonfAI deep learning framework for medical imaging.</p>"
            "<p>If you use KonfAI in your research, please cite the following work:<br>"
            "Boussot V., Dillenseger J.-L.:<br>"
            "<b>KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.</b><br>"
            '<a href="https://arxiv.org/abs/2508.09823">https://arxiv.org/abs/2508.09823</a>'
            "</p>"
        )




def _is_reload_setup(moduleName: str) -> bool:
    key = f"{moduleName}.wasSetupOnce"
    was = bool(slicer.app.property(key))
    slicer.app.setProperty(key, True)
    return was


class KonfAIWidget(ScriptedLoadableModuleWidget):
    """
    Top-level scripted loadable module widget for KonfAI.

    This class ties together the Slicer module system with the KonfAICoreWidget,
    which handles actual application logic and GUI.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().__init__(parent)

    def setup(self) -> None:
        """
        Construct and initialize the KonfAI module GUI.

        This method is called once when the user first opens the module.
        """
        super().setup()
        # Create the core KonfAI widget
        self.konfai_core = KonfAICoreWidget("KonfAI Apps")

        # Create and register one KonfAI app specialized for inference
        prediction_widget = KonfAIAppTemplateWidget(
            "Inference",
            ["VBoussot/ImpactSynth", "VBoussot/MRSegmentator-KonfAI", "VBoussot/TotalSegmentator-KonfAI", "VBoussot/ImpactSeg"],
        )
        self.konfai_core.register_apps([prediction_widget])

        # Attach the core widget to the Slicer module layout
        self.layout.addWidget(self.konfai_core)

        if _is_reload_setup("SlicerKonfAI"):
            self.konfai_core.enter()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.

        This hook can be used to ensure state is up-to-date when the user
        returns to the module. Currently no additional logic is required.
        """
        self.konfai_core.enter()

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.

        This hook can be used to pause or finalize ongoing tasks, but
        no special handling is required at the moment.
        """
        self.konfai_core.exit()


class KonfAITest(ScriptedLoadableModuleTest):
    """
    Minimal self-test backing the `slicer_add_python_unittest(SCRIPT KonfAI.py)`
    registration. The full API contract used by ImpactSynth/ImpactReg is
    checked by Testing/Python/KonfAIApiContractTest.py.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_public_api_facade()

    def test_public_api_facade(self):
        """The symbols imported by the sister extensions must exist."""
        import KonfAI as konfai_module

        for symbol in (
            "AppTemplateWidget",
            "ChipSelector",
            "KonfAIAppTemplateWidget",
            "KonfAIAppInferencePanel",
            "KonfAIAppQAPanel",
            "KonfAIAppSelectionPanel",
            "KonfAICoreWidget",
            "KonfAIMetricsPanel",
            "Process",
            "RemoteServer",
            "_is_reload_setup",
            "has_node_content",
            "KONFAI_SLICER_API_VERSION",
        ):
            self.assertTrue(hasattr(konfai_module, symbol), f"missing public symbol: {symbol}")
