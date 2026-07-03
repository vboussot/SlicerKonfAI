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

"""Reusable combo-to-chips selector (ensemble checkpoints)."""

from collections.abc import Callable

from qt import QPushButton


class ChipSelector:
    """
    Generic "ComboBox -> removable chips" controller.

    Responsibilities:
      - Add a chip when user selects an item from combo
      - Remove chip on click, and put the item back in combo
      - Keep an optional spinbox in sync with number of selected chips
      - Notify optional callback on change

    Assumptions:
      - `container_layout` is a QLayout that may end with a spacer item
      - combo exposes itemText(i), addItem(text), removeItem(i), findText(text),
        and emits activated(int) (Qt signal)
    """

    def __init__(
        self,
        combo,
        container_layout,
        spinbox=None,
        min_selected: int = 0,
        combo_remove: bool = True,
        on_change: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._combo = combo
        self._layout = container_layout
        self._spinbox = spinbox
        self._min_selected = max(0, int(min_selected))
        self._on_change = on_change
        self._combo_remove = combo_remove
        self._combo.connect("activated(int)", self._on_combo_activated)
        if self._spinbox is not None:
            self._spinbox.setMinimum(self._min_selected)
            self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        self._all_items: list[str] = []
        self._available_items: list[str] = []

    def _on_spinbox_changed(self):
        sel = self.selected()
        if self._spinbox.value == len(sel):
            return
        if self._spinbox.value > len(sel):
            self._add(sorted(set(self._all_items) - set(sel))[0])
        else:
            text = sel[-1]
            self._remove(text)

    def update(self, availables_item: list[str], all_items: list[str], pre_selected: list[str]):
        self._all_items = all_items
        self._available_items = availables_item
        self._combo.clear()
        if self._spinbox is not None:
            self._spinbox.setMaximum(len(all_items))
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, QPushButton):
                self._layout.removeWidget(widget)
                widget.deleteLater()

        for name in all_items:
            if name in pre_selected:
                self._add(name)
            else:
                self._combo.addItem(name)
        if len(self.selected()) == 0 and len(all_items) > 0:
            self._add(all_items[0])

    def selected(self) -> list[str]:
        """Return currently selected chip texts (in layout order)."""
        out: list[str] = []
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                out.append(w.text)
        return out

    def _add(self, text: str) -> None:
        """Add a chip (if not already selected) and remove it from combo."""
        if not text or text in self.selected():
            return

        btn = QPushButton(text)
        btn.flat = True
        btn.toolTip = f"Click to remove {text}"
        btn.minimumHeight = 20
        btn.maximumHeight = 24

        shadow = text not in self._available_items
        btn.setProperty("shadow", "1" if shadow else "0")
        btn.setStyleSheet("""
            QPushButton {
                color: #0b3d91;
                background-color: #edf3ff;
                border: 1px solid #0b3d91;
                border-radius: 12px;
                padding: 3px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #dce8ff; }

            QPushButton[shadow="1"] {
                color: #6b7280;
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                font-weight: 500;
            }
            QPushButton[shadow="1"]:hover { background-color: #e5e7eb; }
            QPushButton[shadow="1"]:pressed { background-color: #e5e7eb; }
            """)

        def _remove():
            self._remove(text)

        btn.clicked.connect(_remove)
        self._layout.insertWidget(self._insert_index_before_spacer(), btn)

        if self._combo_remove:
            idx = self._combo.findText(text)
            if idx != -1:
                self._combo.removeItem(idx)
        self._sync()

    def _remove(self, text: str) -> None:
        """Remove a chip (respecting min_selected) and add it back to combo."""
        if not text:
            return

        cur = self.selected()
        if len(cur) <= self._min_selected:
            return

        # remove chip widget
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if isinstance(w, QPushButton) and w.text == text:
                self._layout.removeWidget(w)
                w.deleteLater()
                break

        # add back to combo (avoid duplicates)
        if self._combo.findText(text) == -1:
            self._combo.addItem(text)

        self._sync()

    def _sync(self) -> None:
        sel = self.selected()
        if self._spinbox is not None:
            try:
                self._spinbox.setValue(len(sel))
            except Exception:
                pass
        if self._on_change is not None:
            self._on_change(sel)

    def _on_combo_activated(self, index: int) -> None:
        text = self._combo.itemText(index)
        self._add(text)

    def _insert_index_before_spacer(self) -> int:
        insert_index = self._layout.count()
        for i in range(self._layout.count()):
            if self._layout.itemAt(i).spacerItem() is not None:
                insert_index = i
                break
        return insert_index
