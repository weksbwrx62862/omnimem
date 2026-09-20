"""utils/experimental.py 装饰器单测（离线）。"""
from __future__ import annotations

import logging

import pytest
from omnimem.utils import experimental as exp


@pytest.fixture(autouse=True)
def _reset_warned():
    exp._warned.clear()
    yield
    exp._warned.clear()


def test_experimental_marks_doc_and_preserves_call():
    @exp.experimental
    def add(a, b):
        """原始文档"""
        return a + b

    assert add.__doc__.startswith("[Experimental]")
    assert add(2, 3) == 5


def test_experimental_warns_only_once(caplog):
    @exp.experimental
    def noop():
        return 1

    with caplog.at_level(logging.WARNING, logger="omnimem.utils.experimental"):
        noop()
        noop()
        noop()
    warns = [r for r in caplog.records if "is experimental" in r.getMessage()]
    assert len(warns) == 1


def test_experimental_class_doc_and_init():
    @exp.experimental_class
    class Thing:
        """类文档"""

        def __init__(self, x):
            self.x = x

    assert Thing.__doc__.startswith("[Experimental]")
    obj = Thing(7)
    assert obj.x == 7


def test_experimental_class_warns_once(caplog):
    @exp.experimental_class
    class Once:
        def __init__(self):
            self.built = True

    with caplog.at_level(logging.WARNING, logger="omnimem.utils.experimental"):
        Once()
        Once()
    warns = [r for r in caplog.records if "is experimental" in r.getMessage()]
    assert len(warns) == 1
