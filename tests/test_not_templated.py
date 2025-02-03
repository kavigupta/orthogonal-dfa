"""
Delete this file once all other tests pass. It will fail if any template tags are used in the code.
It will fail if it exists, because it includes the template tag "_OF_PACKAGE" which should not be used.
"""

import os
import unittest

from parameterized import parameterized


def allFiles():
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            yield False, os.path.join(root, dir)
        for file in files:
            yield True, os.path.join(root, file)


class TestTemplateTagsNoLongerExist(unittest.TestCase):

    @parameterized.expand(
        [(file_path,) for is_file, file_path in allFiles() if is_file]
    )
    def test_template_tags_no_longer_exist(self, file):
        """
        Go through all files in this directory and ensure that no template tags are used.
        """
        try:
            with open(file) as f:
                contents = f.read()
        except UnicodeDecodeError:
            return
        self.assertNotIn("_OF_PACKAGE", contents)

    @parameterized.expand([(file_path,) for _, file_path in allFiles()])
    def test_no_template_tags_in_names(self, file):
        """
        Go through all files in this directory and ensure that no template tags are used in names.
        """
        self.assertNotIn("_OF_PACKAGE", file)
