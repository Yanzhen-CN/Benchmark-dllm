import argparse

import pytest

from compare_sudoku_traces import parse_model_spec


def test_parse_model_spec_keeps_architecture_and_config_axes_separate():
    assert parse_model_spec("llada2_1:qmode") == ("llada2_1", "qmode")


def test_parse_model_spec_rejects_missing_config():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_spec("llada2_1")
