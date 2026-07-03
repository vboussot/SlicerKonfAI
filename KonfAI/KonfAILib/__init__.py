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

"""Shared library of the KonfAI Slicer extension.

Layout:
- ``logic/``: no direct access to widgets; UI feedback goes through
  injected callbacks.
- ``widgets/``: Qt/MRML widgets and dialogs composing the module GUI.

Rules:
- This package is distributed only by the KonfAI extension. Sister
  extensions (ImpactSynth, ImpactReg) must import from the ``KonfAI``
  module facade, never from ``KonfAILib`` directly.
- No module-level import of ``konfai``, ``konfai_apps`` or ``keyring``
  anywhere in this package: those are installed on demand after Slicer
  startup, so they must only be imported inside functions.
- Keep every ``__init__.py`` in this package free of imports so that
  loading the facade stays cheap and side-effect free.

API versioning (MAJOR, MINOR): sister extensions require
``major == N and minor >= m``. MAJOR is only incremented after a
deprecation cycle of one release during which removed symbols and
changed signatures keep working through aliases.
"""

KONFAI_SLICER_API_VERSION = (2, 0)
