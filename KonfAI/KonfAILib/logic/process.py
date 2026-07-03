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

"""QProcess wrapper running konfai-apps CLI commands with log/progress forwarding."""

import re
import time
from collections.abc import Callable
from pathlib import Path

from qt import QProcess

# tqdm renders nested/multi-line progress bars with ANSI cursor-control sequences (e.g. ``\x1b[A`` to move
# up). Terminals interpret them invisibly, but a GUI log panel shows them verbatim (``[A``). Strip every CSI
# escape so forwarded log lines stay clean without affecting the percentage/speed parsing below.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Process(QProcess):
    """
    Thin wrapper around QProcess to handle KonfAI CLI processes.

    Responsibilities:
      - Forward stdout lines to the GUI log callback
      - Parse progress and speed from stdout and update the progress bar
      - Forward stderr lines to the console for debugging
    """

    def __init__(
        self,
        update_logs: Callable[[str, bool | None], None],
        update_progress: Callable[[int, str | None | None], None],
        running_setter: Callable[[bool], None],
    ):
        super().__init__(self)
        self.readyReadStandardOutput.connect(self.on_stdout_ready)
        self.readyReadStandardError.connect(self.on_stderr_ready)

        # Callbacks defined by the KonfAI core widget
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._running_setter = running_setter

    def on_stdout_ready(self) -> None:
        """
        Handle new data available on the standard output.

        This method:
          - Normalizes line endings
          - Forwards the message to the log window
          - Extracts numerical progress (0–100 %) and speed (e.g., '5 it/s')
            from the log line when present, and forwards them to the UI.
        """
        line = _ANSI_RE.sub("", self.readAllStandardOutput().data().decode()).strip()
        if line:
            # Keep only the last sub-line if carriage returns are used for progress overwrite
            line = line.replace("\r\n", "\n").split("\r")[-1]

            # Forward to the log callback
            self._update_logs(line, False)

            # Parse progress percentage if present (e.g., " 45% 123/456")
            m = re.search(r"\b(\d{1,3})%(?=\s+\d+/\d+)", line)
            # Parse speed pattern if present (e.g., "5.2 it/s" or "0.21 s/it")
            speed = re.search(r"([\d.]+)\s*(it/s|s/it)", line)

            if m:
                pct = int(m.group(1))
            else:
                pct = None

            if speed is not None and pct is not None:
                # Notify UI of updated progress and speed
                self._update_progress(pct, speed.group(1) + " " + speed.group(2))
            else:
                m = re.search(r"\b(\d{1,3})%\|", line)
                if m:
                    pct = int(m.group(1))
                    self._update_progress(pct, "")

    def on_stderr_ready(self) -> None:
        """
        Handle new data available on the standard error stream.

        tqdm-style progress (library downloads, registration iterations) is emitted on stderr:
        its percentage drives the progress bar, while genuine error lines are forwarded to the
        module log panel so failures (e.g. an evaluation crash) stay visible from the GUI.
        """
        data = _ANSI_RE.sub("", self.readAllStandardError().data().decode())
        if not data.strip():
            return
        for chunk in data.replace("\r\n", "\n").split("\n"):
            # tqdm overwrites a line with carriage returns; keep only its latest state.
            line = chunk.split("\r")[-1].strip()
            if not line:
                continue
            # tqdm progress is emitted on stderr and refreshed many times per second. Depending on ncols it
            # renders as "45%|███|" or, with ncols=0, as a bare "45% 3/25 [.. it/s]"; a plain download shows
            # only a rate. Any percentage-with-count / pipe bar or rate line is progress: use it to drive the
            # bar and never append it to the log — otherwise each of the ~thousand refreshes piles up as a
            # duplicated line in the panel.
            percent = re.search(r"\b(\d{1,3})%(?:\s+\d+/\d+|\s*\|)", line)
            speed = re.search(r"([\d.]+\s*(?:[KMGT]?B/s|it/s|s/it))", line)
            if percent or speed:
                if percent:
                    self._update_progress(int(percent.group(1)), speed.group(1) if speed else "")
                continue
            self._update_logs(f"[stderr] {line}", False)

    def run(self, command: str, work_dir: Path, args: list[str], on_end_function: Callable[[], None]) -> None:
        """
        Start a new subprocess with the given command and arguments.

        Parameters
        ----------
        command : str
            Executable name (e.g., 'konfai-apps').
        work_dir : Path
            Working directory in which the process will be executed.
        args : list[str]
            List of command-line arguments to pass to the executable.
        on_end_function : callable
            Callback to execute once the process has finished.
        """
        self.setWorkingDirectory(str(work_dir))

        t0 = time.perf_counter()

        # Disconnect any previous 'finished' slot to avoid stacking connections
        try:
            self.finished.disconnect()
        except TypeError:
            # No slot connected yet
            pass

        def _on_finished() -> None:
            if self.exitStatus() != QProcess.NormalExit:
                self._running_setter(False)
                return
            on_end_function()
            dt = time.perf_counter() - t0
            h, r = divmod(dt, 3600)
            m, s = divmod(r, 60)
            if h >= 1:
                msg = f"{int(h)}h {int(m)}m {s:.1f}s."
            elif m >= 1:
                msg = f"{int(m)}m {s:.1f}s."
            else:
                msg = f"{s:.2f}s."

            self._update_logs(f"Processing finished in {msg}", False)
            self._update_progress(100, None)
            self._running_setter(False)

        self.finished.connect(_on_finished)

        # Start the process asynchronously
        self.start(command, args)

    def stop(self) -> None:
        """
        Immediately terminate the running process, if any.
        """
        if self.state() == QProcess.ProcessState.NotRunning:
            return
        self.terminate()
        if not self.waitForFinished(2000):
            self.kill()
            self.waitForFinished(-1)
