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

"""Installation and version checks of torch and konfai-apps in Slicer's Python."""

import json
import urllib.request

import slicer


def get_latest_pypi_version(package: str, timeout_s: float = 5.0) -> str | None:
    """
    Return the latest *released* version on PyPI for `package`, or None if unreachable.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["info"]["version"]
    except Exception as exc:
        print(f"[KonfAI] Could not query PyPI for {package}: {exc}")
        return None


def get_installed_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def is_editable_install(package: str) -> bool:
    """
    Return True if `package` is installed in editable/development mode (``pip install -e``).

    A developer working from a local checkout owns their environment; SlicerKonfAI must not
    downgrade or reinstall over an editable install (which would clobber unreleased changes).
    """
    try:
        from importlib.metadata import distribution

        raw = distribution(package).read_text("direct_url.json")
        if not raw:
            return False
        return bool(json.loads(raw).get("dir_info", {}).get("editable", False))
    except Exception:
        return False


def install_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        msg = (
            "SlicerKonfAI requires PyTorch to be installed in 3D Slicer using the official SlicerPyTorch extension.\n\n"
            "Step 1 – Install the SlicerPyTorch extension:\n"
            "  • In Slicer, open:  View → Extensions Manager\n"
            "  • Search for:  PyTorch (or SlicerPyTorch)\n"
            "  • Install the extension and restart Slicer when prompted.\n\n"
            "Step 2 – Install PyTorch using the SlicerPyTorch module:\n"
            "  • Open:  View → Modules\n"
            "  • Select:  Utilities → PyTorch\n"
            "  • Click 'Install PyTorch and choose the CPU or GPU build\n"
            "    that matches your system configuration.\n\n"
        )

        slicer.util.infoDisplay(msg, windowTitle="Prerequisite: PyTorch installation required")
        slicer.util.selectModule("Data")
        return False


def install_package(package: str, display_name: str) -> bool:
    """Ensure ``package`` (pinned to its latest PyPI release) is installed in Slicer's Python.

    An editable/development install is respected and never downgraded. Returns True when the package is
    available (installed now or already present), False when the user declined a required install. This is
    the shared installer used by every KonfAI-based extension (KonfAI, IMPACT-Reg, …); the released
    ``package`` pins its own matching ``konfai``/``konfai-apps`` versions, so installing it pulls the stack.
    """
    # Deferred import: keeps the logic package importable without pulling Qt widgets at module load.
    from KonfAILib.widgets.helpers import ask_user_to_install_dependency, slicer_wait_popup

    installed = get_installed_version(package)

    if installed is not None and is_editable_install(package):
        print(f"[{display_name}] editable install detected ({installed}) -> respecting development environment.")
        return True

    latest = get_latest_pypi_version(package)

    if latest is None:
        if installed is not None:
            print(f"[{display_name}] installed ({installed}), PyPI unreachable -> keeping current.")
            return True
        if not ask_user_to_install_dependency(
            display_name,
            f"{display_name} is not installed.\n"
            "PyPI cannot be reached right now, but we can still try installing.\n"
            "Do you want to try?",
        ):
            slicer.util.selectModule("Data")
            return False
        with slicer_wait_popup(f"{display_name} dependency install", f"Installing {package}..."):
            slicer.util.pip_install(package)
        return True

    if installed is None:
        if not ask_user_to_install_dependency(
            display_name,
            f"{display_name} is required.\nLatest available version on PyPI: {latest}\n\n"
            "Do you want to install it now?",
        ):
            slicer.util.selectModule("Data")
            return False
        with slicer_wait_popup(f"{display_name} dependency install", f"Installing {package} {latest}..."):
            slicer.util.pip_install(f"{package}=={latest}")
        return True

    if installed != latest:
        if not ask_user_to_install_dependency(
            display_name,
            f"A newer {display_name} version is available.\n"
            f"Installed: {installed}\n"
            f"Latest:    {latest}\n\n"
            "Do you want to upgrade now?",
        ):
            return True
        with slicer_wait_popup(
            f"{display_name} dependency upgrade",
            f"Upgrading {package} from {installed} to {latest}...\nThis may take several minutes.\n\n",
        ):
            slicer.util.pip_install(f"{package}=={latest}")
    return True


def install_konfai() -> bool:
    if not install_package("konfai-apps", "KonfAI"):
        return False
    # A development checkout is trusted as-is (mirrors install_package's editable short-circuit).
    if is_editable_install("konfai-apps"):
        return True

    # Deferred import: keeps the logic package importable without pulling Qt widgets at module load.
    from KonfAILib.widgets.helpers import ask_user_to_install_dependency, slicer_wait_popup

    import konfai

    try:
        konfai.assert_konfai_install()
    except Exception as e:
        latest = get_latest_pypi_version("konfai-apps")
        if not ask_user_to_install_dependency(
            "KonfAI",
            "KonfAI-apps is installed but not functional (missing/broken dependencies).\n\n"
            "Do you want to reinstall KonfAI-apps (and its dependencies) now?\n\n"
            f"Details:\n{e}",
        ):
            slicer.util.selectModule("Data")
            return False
        with slicer_wait_popup("KonfAI-apps repair", "Reinstalling KonfAI-apps and dependencies..."):
            slicer.util.pip_install(f"--upgrade --force-reinstall --no-deps konfai-apps=={latest}")
            slicer.util.pip_install(
                "tqdm numpy ruamel.yaml psutil tensorboard lxml h5py nvidia-ml-py requests huggingface_hub"
            )
        try:
            konfai.assert_konfai_install()
        except Exception as e2:
            slicer.util.errorDisplay("KonfAI-apps was reinstalled but is still not functional.\n\n" f"{e2}")
            slicer.util.selectModule("Data")
            return False
    return True
