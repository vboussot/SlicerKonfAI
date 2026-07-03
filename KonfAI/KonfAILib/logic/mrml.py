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

"""MRML node predicates and helpers."""


def has_node_content(node) -> bool:
    if not node:
        return False

    if node.IsA("vtkMRMLVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode"):
        return bool(node.GetImageData())

    if node.IsA("vtkMRMLSegmentationNode"):
        seg = node.GetSegmentation()
        return seg is not None and seg.GetNumberOfSegments() > 0

    return False
