import sys
from unittest.mock import MagicMock

import importlib


def run_main_with_args(args):
    # reload to ensure argparse parses new sys.argv
    importlib.reload(importlib.import_module('src.train'))
    sys_argv = ["src.train"] + args
    return sys_argv


def test_cli_dispatches_logistic(monkeypatch, capsys):
    # Patch train_logistic to capture calls
    import src.train as train_mod
    mocked = MagicMock()
    monkeypatch.setattr("src.train_logistic.train_logistic", mocked)

    argv = run_main_with_args(["--model", "logistic", "--experiment", "audio", "--save-model", "false"])
    monkeypatch.setattr(sys, "argv", argv)

    train_mod.main()

    mocked.assert_called_once()
    # verify it was called with expected keyword args by inspecting the first call
    called_args, called_kwargs = mocked.call_args
    assert called_kwargs.get("experiment") in ("audio", "Audio", "AUDIO") or called_args == ("audio", False) or called_args[0] == "audio"


def test_cli_dispatches_lgbm(monkeypatch):
    import src.train as train_mod
    mocked = MagicMock()
    monkeypatch.setattr("src.train_lgbm.train_lgbm", mocked)

    argv = run_main_with_args(["--model", "lightgbm", "--experiment", "genre", "--save-model", "true"])
    monkeypatch.setattr(sys, "argv", argv)

    train_mod.main()

    mocked.assert_called_once()
