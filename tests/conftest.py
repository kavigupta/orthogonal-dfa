"""What every test run here shares.

`setup.cfg` sets one clock for the whole suite, which has to suit the tests that
share a shard.  A slow test has a shard and a job of its own, so it gets the
clock that job allows instead.
"""

import pytest

#: Under the 40 minutes the slow shards are given, leaving room for the install.
SLOW_TIMEOUT_SECONDS = 2100


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.timeout(SLOW_TIMEOUT_SECONDS))
