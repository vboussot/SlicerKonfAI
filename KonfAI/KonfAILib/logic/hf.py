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

"""Hugging Face repository helpers (to move into konfai-apps eventually)."""

from pathlib import PurePosixPath


def split_hf_repo_reference(repo_id: str) -> tuple[str, str | None]:
    """
    Split a Hugging Face repository reference into repo id and optional revision.
    """
    base_repo_id, _, revision = repo_id.partition("@")
    return base_repo_id, revision or None


def get_hf_app_file_list(repo_id: str, app_name: str) -> list[str]:
    """
    Return all files contained in a Hugging Face app folder, recursively.

    The returned paths are relative to the app folder so they can be displayed
    clearly in the UI and re-expanded into allow_patterns when syncing.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    base_repo_id, revision = split_hf_repo_reference(repo_id)

    try:
        tree = HfApi().list_repo_tree(
            repo_id=base_repo_id,
            path_in_repo=app_name,
            recursive=True,
            revision=revision,
            repo_type="model",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect '{app_name}' in Hugging Face repository '{repo_id}'.\n{exc}") from exc

    app_root = PurePosixPath(app_name)
    files: list[str] = []
    for entry in tree:
        if isinstance(entry, RepoFolder):
            continue
        try:
            relative_path = PurePosixPath(entry.path).relative_to(app_root)
        except ValueError:
            continue
        files.append(str(relative_path))

    return sorted(files)
