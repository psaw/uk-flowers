# %% IMPORTS

import sys
from unittest import mock
from pathlib import Path

import pytest
from omegaconf import DictConfig

from uk_flowers import scripts

# %% FUNCTIONS


def _captured_cfg(positional_args: tuple[object, ...]) -> DictConfig:
    """
    Extract the composed Hydra config from a mocked pipeline call.

    `scripts.run_train`/`scripts.run_infer` call ``train(cfg)``/``infer(cfg)``
    (see `uk_flowers/scripts.py`); the imported signatures are
    ``train(cfg: DictConfig) -> None`` and ``infer(cfg: DictConfig) -> None``
    (see `uk_flowers/train.py` and `uk_flowers/infer.py`). Hence the single
    positional argument of the call is the `DictConfig` composed by
    ``hydra.compose``.

    Args:
        positional_args: Positional arguments of the mocked call.

    Returns:
        The composed Hydra configuration.
    """
    assert len(positional_args) == 1, "Expected exactly one positional argument (the cfg)."
    cfg = positional_args[0]
    assert isinstance(cfg, DictConfig), "The first positional argument must be the Hydra cfg."
    return cfg


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
    cfg = _captured_cfg(mock_train.call_args.args)
    assert cfg.train.epochs == 7
    assert cfg.project.seed == 42


def test_main_dispatches_infer_command_with_overrides(tmp_path: Path) -> None:
    # given
    image_path = tmp_path / "photo.jpg"
    with (
        mock.patch("uk_flowers.scripts.infer") as mock_infer,
        mock.patch.object(
            sys, "argv", ["uk-flowers", "infer", f"inference.image_path={image_path}"]
        ),
    ):
        # when
        scripts.main()
    # then
    mock_infer.assert_called_once()
    cfg = _captured_cfg(mock_infer.call_args.args)
    assert Path(cfg.inference.image_path) == image_path


def test_run_train_passes_hydra_overrides_to_train() -> None:
    # given
    overrides = ["train.epochs=3", "project.seed=7", "model.pretrained=false"]
    with mock.patch("uk_flowers.scripts.train") as mock_train:
        # when
        scripts.run_train(overrides)
    # then
    mock_train.assert_called_once()
    cfg = _captured_cfg(mock_train.call_args.args)
    assert cfg.train.epochs == 3
    assert cfg.project.seed == 7
    assert cfg.model.pretrained is False


def test_run_infer_passes_hydra_overrides_to_infer(tmp_path: Path) -> None:
    # given
    image_path = tmp_path / "photo.jpg"
    output_path = tmp_path / "result.json"
    overrides = [f"inference.image_path={image_path}", f"inference.output_path={output_path}"]
    with mock.patch("uk_flowers.scripts.infer") as mock_infer:
        # when
        scripts.run_infer(overrides)
    # then
    mock_infer.assert_called_once()
    cfg = _captured_cfg(mock_infer.call_args.args)
    assert Path(cfg.inference.image_path) == image_path
    assert Path(cfg.inference.output_path) == output_path


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
