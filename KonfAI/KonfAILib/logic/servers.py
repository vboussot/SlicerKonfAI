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

import json
import ssl

import slicer
from qt import QSettings

SERVICE = "KonfAI"
#: Where each server's host, port and transport are kept between sessions; its token is in the keyring.
SETTINGS_KEY = "KonfAI-Settings/RemoteServers"


def stored_servers() -> dict[str, dict]:
    """The servers kept in the settings, by name."""
    raw = QSettings().value(SETTINGS_KEY)
    return json.loads(raw) if raw else {}


def store_server(remote_server: "RemoteServer") -> None:
    """Keep ``remote_server``'s host, port and transport in the settings, under its name."""
    servers = stored_servers()
    servers[remote_server.name] = {
        "host": remote_server.host,
        "port": remote_server.port,
        "https": remote_server.https,
        "ca_cert": remote_server.ca_cert,
    }
    QSettings().setValue(SETTINGS_KEY, json.dumps(servers))


class RemoteServer:

    def __init__(self, name: str, host: str, port: int, https: bool | None = None, ca_cert: str | None = None) -> None:
        self.name = name
        self.host = host
        self.port = port
        # Rebuilt from name, host and port alone, a server takes its transport from the settings.
        stored = stored_servers().get(name, {}) if https is None else {}
        self.https = bool(stored.get("https", False)) if https is None else https
        self.ca_cert = (stored.get("ca_cert") or None) if https is None else (ca_cert or None)
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
        return f"{'https' if self.https else 'http'}://{self.host}:{self.port}"

    def ssl_context(self) -> ssl.SSLContext | None:
        """The context an HTTPS server's certificate is checked in: ``ca_cert``, else the system's authorities."""
        return ssl.create_default_context(cafile=self.ca_cert) if self.https else None

    def cli_args(self) -> list[str]:
        """The options that send a konfai-apps command to this server."""
        args = ["--host", self.host, "--port", str(self.port)]
        if self.token:
            args += ["--token", self.token]
        if self.ca_cert:
            args += ["--ca-cert", self.ca_cert]
        elif self.https:
            args.append("--https")
        return args

    def get_json(self, path: str, timeout_s: float, params: list[tuple[str, int]] | None = None) -> dict:
        """The JSON body of ``GET <base url>/<path>``, as konfai's own RemoteServer provides it
        (konfai.get_ram, get_vram and get_available_devices call this on the server object)."""
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        url = f"{self.get_url()}/{path}"
        if params:
            url += "?" + urlencode(params)
        with urlopen(  # nosec B310
            Request(url, headers=self.get_headers()), timeout=timeout_s, context=self.ssl_context()
        ) as response:
            return json.loads(response.read().decode("utf-8"))
