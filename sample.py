import jax
from typing import Tuple
from jax.lax import map
import jax.random as jr
import jax.numpy as jnp

from trainer.ddpm_trainer import DiffusionModelSchedule, diffusion_sample

from model.unet import Unet_CelebA
from data_loader.celeba import CelebAInfo

from core.serialization import load_ivon_state, load_model_params

from utils import load_diffusion_ivon_config


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="seed for initializing the model",
    )

    parser.add_argument(
        "--samples-seed",
        type=int,
        default=42,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        default=700,
        help="seed for samples generating",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=16,
        help="Number of MC samples for uncertainty estimation",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="A .msgpack file with the model for Unet",
    )

    parser.add_argument(
        "--opt-state",
        type=str,
        required=True,
        help="A .msgpack file with the opt state",
    )

    parser.add_argument(
        "--samples-output-dir",
        required=True,
        type=str,
        help="A dir for the samples output",
    )

    args = parser.parse_args()

    config = load_diffusion_ivon_config()

    unet = Unet_CelebA()

    unet_vars = unet.init(
        jr.key(args.init_seed), jnp.zeros((1, *CelebAInfo.shape)), jnp.zeros((1,))
    )

    mean_params = load_model_params(args.model, unet_vars["params"])
    ivon_state = load_ivon_state(args.opt_state, unet_vars["params"])

    diffusion = DiffusionModelSchedule.create(timesteps=config["diffusion_timesteps"])

    samples = diffusion_sample(
        model=unet,
        key=jr.key(1),
        variables={"params": mean_params},
        shape=(32, *CelebAInfo.shape),
        schedule=diffusion,
    )

    jnp.save("celeba_samples.npy", samples)
