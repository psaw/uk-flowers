# %% IMPORTS

import sys
from unittest import mock

import pytest

from uk_flowers import scripts

# %% FUNCTIONS


def test_main_dispatches_train_command_with_overrides() -> None:
    # given
    with (
        mock.patch("uk_flowers.scripts.train") as mock_train,
        mock.patch.object(sys, "argv", ["uk-flowers", "train", "train.epochs=7"]),
    ):
        # when
        scripts.main()
    # then
    mock_train.assert_called_once()
    cfg = mock_train.call_args.args[0]
    assert cfg.train.epochs == 7
    assert cfg.project.seed == 42


def test_main_dispatches_infer_command_with_overrides() -> None:
    # given
    with (
        mock.patch("uk_flowers.scripts.infer") as mock_infer,
        mock.patch.object(
            sys, "argv", ["uk-flowers", "infer", "inference.image_path=/tmp/photo.jpg"]
        ),
    ):
        # when
        scripts.main()
    # then
    mock_infer.assert_called_once()
    cfg = mock_infer.call_args.args[0]
    assert cfg.inference.image_path == "/tmp/photo.jpg"


def test_run_train_passes_hydra_overrides_to_train() -> None:
    # given
    overrides = ["train.epochs=3", "project.seed=7", "model.pretrained=false"]
    with mock.patch("uk_flowers.scripts.train") as mock_train:
        # when
        scripts.run_train(overrides)
    # then
    mock_train.assert_called_once()
    cfg = mock_train.call_args.args[0]
    assert cfg.train.epochs == 3
    assert cfg.project.seed == 7
    assert cfg.model.pretrained is False


def test_run_infer_passes_hydra_overrides_to_infer() -> None:
    # given
    overrides = ["inference.image_path=/tmp/photo.jpg", "inference.output_path=/tmp/result.json"]
    with mock.patch("uk_flowers.scripts.infer") as mock_infer:
        # when
        scripts.run_infer(overrides)
    # then
    mock_infer.assert_called_once()
    cfg = mock_infer.call_args.args[0]
    assert cfg.inference.image_path == "/tmp/photo.jpg"
    assert cfg.inference.output_path == "/tmp/result.json"


def test_run_infer_reports_missing_image_path_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    # given
    with mock.patch(
        "uk_flowers.scripts.infer",
        side_effect=ValueError("Option cfg.inference.image_path must be provided."),
    ):
        # when
        with pytest.raises(SystemExit) as exc_info:
            scripts.run_infer([])
    # then
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Option cfg.inference.image_path must be provided." in captured.err
