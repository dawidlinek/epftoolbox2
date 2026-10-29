"""YAML writing shared by the pipelines' ``save()`` methods."""

from typing import Any

import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    """A ``SafeDumper`` that writes an object used twice (a predictor list shared by two models) in full each time,
    instead of as an anchor and alias (``&id001`` / ``*id001``). The aliases are valid YAML, but they make a saved
    configuration hard to read and to edit, since changing one model's list would silently change the other's."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def dump_yaml(data: Any) -> str:
    """``data`` as block-style YAML text with the keys in their given order and no anchors or aliases.

    Returns the text instead of writing to a stream, so a value YAML cannot represent raises before the caller
    opens (and truncates) the file it was going to replace.
    """
    return yaml.dump(data, Dumper=NoAliasSafeDumper, default_flow_style=False, sort_keys=False)
