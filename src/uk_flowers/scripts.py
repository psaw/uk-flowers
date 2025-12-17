"""Scripts of the project."""

# %% IMPORTS

import sys
import fire
import hydra

from uk_flowers.train import train
from uk_flowers.infer import infer

# %% FUNCTIONS


def run_train(overrides: list[str] | None = None) -> None:
    """
    Run training pipeline.

    Args:
        overrides: List of Hydra overrides (e.g. ["training.epochs=10"]).
    """
    # Initialize Hydra and compose configuration
    with hydra.initialize(version_base=None, config_path="../../confs"):
        if isinstance(overrides, str):
            overrides = [overrides]
        elif isinstance(overrides, tuple):
            overrides = list(overrides)

        cfg = hydra.compose(config_name="config", overrides=overrides or [])
        train(cfg)


def run_infer(overrides: list[str] | None = None) -> None:
    """
    Run inference pipeline.

    Args:
        overrides: List of Hydra overrides (e.g. ["inference.image_path=..."]).
    """
    # Initialize Hydra and compose configuration
    with hydra.initialize(version_base=None, config_path="../../confs"):
        # Fire passes arguments as a tuple if multiple arguments are provided, or a string if single.
        # If overrides is a string, we need to wrap it in a list.
        # Also, Fire might split arguments by spaces if not careful, but here we expect a list of strings.
        if isinstance(overrides, str):
            overrides = [overrides]
        elif isinstance(overrides, tuple):
            overrides = list(overrides)

        cfg = hydra.compose(config_name="config", overrides=overrides or [])
        try:
            infer(cfg)
        except ValueError as e:
            if "cfg.inference.image_path must be provided" in str(e):
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            raise


def main() -> None:
    """Run the main script function."""
    fire.Fire(
        {
            "train": run_train,
            "infer": run_infer,
        }
    )


if __name__ == "__main__":
    main()
