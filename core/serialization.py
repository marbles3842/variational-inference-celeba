import jax
from flax import serialization
from flax.core import FrozenDict
from .ivon import empty_ivon_state, IvonState


def save_params(path: str, params: FrozenDict | IvonState):
    """Save model parameters or optimizer state to disk.

    Args:
        path: File path to save to
        params: Parameters or state to save
    """
    bytes_output = serialization.to_bytes(params)

    with open(path, "wb") as f:
        f.write(bytes_output)


def load_model_params(path: str, params: FrozenDict):
    """Load model parameters from disk.

    Args:
        path: File path to load from
        params: Template parameters with correct structure

    Returns:
        Loaded parameters
    """
    with open(path, "rb") as f:
        opt_params = serialization.from_bytes(params, f.read())

    return opt_params


def load_ivon_state(path: str, params: FrozenDict):
    """Load IVON optimizer state from disk.

    Args:
        path: File path to load from
        params: Template parameters for creating state structure

    Returns:
        Loaded IVON state
    """
    target = empty_ivon_state(param=params)
    with open(path, "rb") as f:
        opt_state = serialization.from_bytes(target, f.read())
    return opt_state
