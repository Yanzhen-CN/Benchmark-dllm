from pathlib import Path

import run_prepare


def test_run_prepare_builds_complete_preparation_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_prepare.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    assert run_prepare.main(["--force-data", "-m", "illada", "dreamreasoner"]) == 0

    assert len(calls) == 4
    assert Path(calls[0][0][1]).name == "root.py"
    assert calls[0][0][-1] == "setup"
    assert Path(calls[1][0][1]).name == "setup_venv.py"
    assert calls[1][0].count("-m") == 2
    assert Path(calls[2][0][1]).name == "prepare_data.py"
    assert "--force" in calls[2][0]
    assert Path(calls[3][0][1]).name == "prepare_model.py"
    assert calls[3][0].count("-m") == 2
    assert all(kwargs["check"] is True for _, kwargs in calls)


def test_run_prepare_can_filter_datasets_and_skip_checkpoint_downloads():
    args = run_prepare.build_parser().parse_args(
        ["-d", "ruler", "hellobench", "--skip-models"]
    )
    commands = run_prepare.build_commands(args)

    assert len(commands) == 3
    data_command = next(command for command in commands if "prepare_data.py" in command[1])
    assert data_command[-3:] == ["-d", "ruler", "hellobench"]
    assert all("prepare_model.py" not in command[1] for command in commands)


def test_run_prepare_dry_run_executes_no_subprocess(monkeypatch):
    monkeypatch.setattr(
        run_prepare.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not execute subprocesses")
        ),
    )
    assert run_prepare.main(["--dry-run"]) == 0
