"""The clock a slow test runs under."""

import unittest
from types import SimpleNamespace

from tests.conftest import SLOW_TIMEOUT_SECONDS, pytest_collection_modifyitems


def _item(*markers):
    held = list(markers)
    return SimpleNamespace(
        added=[],
        get_closest_marker=lambda name: name if name in held else None,
        add_marker=lambda mark: None,
    )


class TestTheClockASlowTestRunsUnder(unittest.TestCase):
    def test_a_slow_test_is_given_the_slow_clock(self):
        added = []
        item = _item("slow")
        item.add_marker = added.append

        pytest_collection_modifyitems([item])

        self.assertEqual([SLOW_TIMEOUT_SECONDS], [mark.args[0] for mark in added])

    def test_every_other_test_keeps_the_suite_wide_one(self):
        added = []
        item = _item()
        item.add_marker = added.append

        pytest_collection_modifyitems([item])

        self.assertEqual([], added)
