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

"""OS integration helpers: open a folder in the system file browser."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import slicer


def _get_system_gui_environment() -> dict[str, str]:
    """
    Build an environment suitable for launching system GUI applications.

    Slicer adjusts library-related variables for its own runtime. External GUI
    apps such as file browsers should instead start from Slicer's startup
    environment, while keeping the current desktop-session variables.
    """
    try:
        env = dict(slicer.util.startupEnvironment())
    except Exception:
        env = dict(os.environ)
        for key in (
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "PYTHONHOME",
            "PYTHONPATH",
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
        ):
            env.pop(key, None)

    for key in (
        "DBUS_SESSION_BUS_ADDRESS",
        "DESKTOP_SESSION",
        "DISPLAY",
        "HOME",
        "KDE_FULL_SESSION",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "USER",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "XDG_CURRENT_DESKTOP",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_DESKTOP",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value

    return env


def _launch_external_gui_command(command: list[str], env: dict[str, str]) -> tuple[bool, str | None]:
    """
    Launch a GUI command and treat fast non-zero exits as failures.

    GUI launchers generally return quickly on success. If the process is still
    alive after a short wait, we consider the launch successful and leave it
    running.
    """
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return False, str(exc)

    try:
        stdout, stderr = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        return True, None

    if process.returncode == 0:
        return True, None

    details = (stderr or stdout or f"Exited with code {process.returncode}").strip()
    return False, details


def open_path_in_file_browser(path: Path | str) -> tuple[bool, str | None]:
    """
    Open a local path in the system file browser.

    Use a clean Slicer startup environment so external launchers do not inherit
    Slicer's bundled libraries.
    """
    local_path = Path(path).expanduser()
    if not local_path.exists():
        return False, f"Path does not exist: {local_path}"

    path_str = str(local_path.resolve())

    if os.name == "nt":
        try:
            os.startfile(path_str)
            return True, None
        except OSError as exc:
            return False, str(exc)

    if sys.platform == "darwin":
        return _launch_external_gui_command(["open", path_str], _get_system_gui_environment())

    if sys.platform.startswith("linux"):
        commands = [["xdg-open", path_str], ["gio", "open", path_str]]
        env = _get_system_gui_environment()
        errors: list[str] = []
        for command in commands:
            if not shutil.which(command[0]):
                continue
            launched, error = _launch_external_gui_command(command, env)
            if launched:
                return True, None
            if error:
                errors.append(f"{command[0]}: {error}")
        return False, "\n".join(errors) if errors else "No working file browser launcher was found."

    return False, f"Unsupported platform for opening a file browser: {sys.platform}"
