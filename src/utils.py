"""Script with helper functions."""
import warnings
import functools
import pathlib
import random

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed: int = 0) -> None:
    """Sets random seed."""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model: torch.nn.Module, args) -> None:
    """Saves model checkpoint.

    Uses torch.save() to save PyTorch models.

    Args:
        model: PyTorch model.
        model_name: Name of policy model.
        args: Parsed arguments.
    """
    model_name = args.model_name
    checkpoint_name = f"{f'{model_name}' if model_name else 'model'}"
    checkpoint_path = "weights"
    dir_path = pathlib.Path(checkpoint_path)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    model_path = dir_path / f"{checkpoint_name}.pth"
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: torch.nn.Module, args) -> None:
    """Loads model from checkpoint.

    Args:
        model: PyTorch model.
        model_name: Name of policy model.
        args: Parsed arguments.
    """
    checkpoint_name = (
        f"{f'{args.model_name}' if args.model_name else 'model'}"
    )
    checkpoint_path = "weights"
    model_path = pathlib.Path(checkpoint_path) / f"{checkpoint_name}.pth"

    if model_path.is_file():
        state_dict = torch.load(f=model_path)
        model.load_state_dict(state_dict=state_dict)
        print(f"\nModel '{checkpoint_name}' loaded.\n")
    else:
        warnings.warn(
            f"\nModel checkpoint '{checkpoint_name}' not found. "
            "Continuing with random weights.\n"
        )


def print_args(args) -> None:
    """Prints parsed arguments to console.

    Args:
        args: Parsed arguments.
    """
    print("\nSnake configuration:\n")
    representation = "{k:.<32}{v}"
    for key, value in vars(args).items():
        print(representation.format(k=key, v=value))
    print("\n")


def eval(function: callable) -> callable:
    """Evaluation decorator for class methods.

    Wraps function that calls a PyTorch module and ensures
    that inference is performed in evaluation model. Returns
    back to training mode after inference.

    Args:
        function: A callable.

    Returns:
        Decorated function.
    """

    @functools.wraps(function)
    def eval_wrapper(self, *args, **kwargs):
        self.eval()
        out = function(self, *args, **kwargs)
        self.train()
        return out

    return eval_wrapper


def save_config(writer: SummaryWriter, args, file_name: str = None) -> None:
    """Saves config in runs folder.

    Args:
        writer: Summary writer class.
        args: Arguments from argparse.
    """
    file_name = file_name if file_name else "config.txt"
    file_path = pathlib.Path(writer.log_dir) / file_name
    representation = "{k:.<32}{v}"
    with open(file_path, "w") as file:
        for key, value in vars(args).items():
            file.write(representation.format(k=key, v=value))
            file.write("\n")
