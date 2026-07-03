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

"""Remote computation servers: value object and keyring-backed token access."""

import slicer

SERVICE = "KonfAI"


class RemoteServer:

    def __init__(self, name: str, host: str, port: int) -> None:
        self.name = name
        self.host = host
        self.port = port
        try:
            import keyring
        except ImportError:
            slicer.util.pip_install("keyring")
        import keyring  # noqa: F811

        self.token = keyring.get_password(SERVICE, str(self))
        self.timeout = 3

    def __str__(self) -> str:
        return f"{self.name}|{self.host}|{self.port}"

    def get_headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"
