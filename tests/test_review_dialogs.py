"""Single-window batch review navigation and Tab preview checks."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.frontend.review_dialogs import BatchReviewDialog  # noqa: E402


class BatchReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_navigation_rejection_and_hold_tab_preview(self) -> None:
        rgb = np.zeros((24, 32, 3), np.uint8)
        overlay = np.zeros((24, 32, 4), np.uint8)

        def load_page(_key):
            return {
                "rgb": rgb, "mask": np.ones((24, 32), bool),
                "raw_overlay": overlay, "icp_overlay": overlay,
                "raw_rmse_mm": 3.0, "icp_rmse_mm": 2.8, "sam_score": 0.9,
                "manual_clicks": [], "positive": [], "negative": [],
                "box_padding": 0.25, "method": "silhouette", "weight": 0.10,
                "error": None,
            }

        dialog = BatchReviewDialog(
            [(0, 1), (0, 2), (1, 1)], load_page,
            rejected={(0, 1): False, (0, 2): False, (1, 1): False},
        )
        dialog.reject_box.setChecked(True)
        dialog._step(1)
        self.assertEqual(dialog._current_key(), (0, 2))
        self.assertTrue(dialog.rejected[(0, 1)])
        self.assertFalse(dialog.rejected[(0, 2)])
        seen = []

        def exercise_tab() -> None:
            self.app.sendEvent(dialog.pose_view, QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier
            ))
            seen.append(dialog._raw_held)
            self.app.sendEvent(dialog.pose_view, QKeyEvent(
                QEvent.Type.KeyRelease, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier
            ))
            seen.append(dialog._raw_held)
            dialog._confirm()

        QTimer.singleShot(0, exercise_tab)
        dialog.exec()
        self.assertEqual(seen, [True, False])
        self.assertTrue(dialog.confirmed)


if __name__ == "__main__":
    unittest.main()
