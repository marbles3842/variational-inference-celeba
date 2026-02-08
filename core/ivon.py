import jax
import jax.numpy as jnp
import optax
import jax.random as jr

from jax.tree_util import tree_map, tree_unflatten, tree_structure

from jax import vmap
from jax.lax import rsqrt, square

from typing import NamedTuple


class _State(NamedTuple):
    current_step: int
    weight_decay: float
    beta1: float
    beta2: float
    hessian_init: float
    ess: float
    avg_nxg: optax.Updates
    momentum: optax.Updates
    hessian: optax.Updates


def _sample_normal_like(key, target_tree):
    treedef = tree_structure(target_tree)
    num_leaves = treedef.num_leaves
    keys = jr.split(key, num_leaves)
    keys_tree = tree_unflatten(treedef, keys)

    return tree_map(
        lambda k, x: jr.normal(k, shape=x.shape, dtype=x.dtype), keys_tree, target_tree
    )


def make_monte_carlo_grad(loss_fn, num_samples):

    def _single_sample(key, mean, std, data):
        noise = _sample_normal_like(key=key, target_tree=mean)
        theta = tree_map(lambda m, s, eps: m + s * eps, mean, std, noise)
        loss, grad = jax.value_and_grad(loss_fn)(theta, data)
        nxg = tree_map(lambda g, s, eps: g * s * eps, grad, std, noise)
        return loss, grad, nxg

    def grad_fn(key, params, opt_state, data):
        keys = jr.split(key, num_samples)

        variance = _compute_variance(opt_state)

        losses, grads, nxgs = vmap(_single_sample, in_axes=(0, None, None, None))(
            keys, params, variance, data
        )

        avg_grad = tree_map(lambda g: jnp.mean(g, axis=0), grads)
        avg_nxg = tree_map(lambda n: jnp.mean(n, axis=0), nxgs)

        return jnp.mean(losses), avg_grad, opt_state._replace(avg_nxg=avg_nxg)

    return grad_fn


def ivon(
    learning_rate: optax.ScalarOrSchedule,
    ess: float,
    weight_decay: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99999,
    hessian_init: float = 1.0,
    clip_radius: float = float("inf"),
    rescale_lr: bool = True,
):
    if weight_decay <= 0:
        raise ValueError("Weight decay must be > 0")

    if hessian_init <= 0:
        raise ValueError("Init value of the Hessian must be > 0")

    transform_list = []

    transform_list.append(
        optax.GradientTransformation(
            init=lambda params: _init_ivon(
                params=params,
                weight_decay=weight_decay,
                beta1=beta1,
                beta2=beta2,
                hessian_init=hessian_init,
                ess=ess,
            ),
            update=_ivon_update,
        )
    )

    if clip_radius < float("inf"):
        transform_list.append(optax.clip(clip_radius))

    transform_list.append(optax.scale_by_learning_rate(learning_rate))

    if rescale_lr:
        transform_list.append(optax.scale(hessian_init + weight_decay))

    return optax.chain(*transform_list)


def _init_ivon(
    params,
    weight_decay: float,
    beta1: float,
    beta2: float,
    hessian_init: float,
    ess: float,
) -> _State:
    momentum = tree_map(jnp.zeros_like, params)
    hessian = tree_map(lambda t: jnp.full_like(t, fill_value=hessian_init), params)
    return _State(
        current_step=1,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        ess=ess,
        avg_nxg=tree_map(jnp.zeros_like, params),
        momentum=momentum,
        hessian=hessian,
        hessian_init=hessian_init,
    )


def _ivon_update(avg_grads: optax.Updates, state: _State, params=None):
    state = _update_hessian(state=state)
    state = _update_momentum(avg_grads=avg_grads, state=state)

    updates = _compute_updates(params=params, state=state)
    return updates, state._replace(current_step=state.current_step + 1)


def _update_hessian(state: _State) -> _State:
    beta2 = state.beta2
    weight_decay = state.weight_decay
    ess = state.ess

    nll_hess = tree_map(
        lambda nxg, h: ess * nxg * (h + weight_decay), state.avg_nxg, state.hessian
    )

    new_hessian = tree_map(
        lambda h, h_hat: beta2 * h
        + (1 - beta2) * h_hat
        + 0.5 * square(1 - beta2) * (h - h_hat) ** 2 / (h + weight_decay),
        state.hessian,
        nll_hess,
    )
    return state._replace(hessian=new_hessian)


def _update_momentum(avg_grads: optax.Updates, state: _State) -> _State:
    g = tree_map(
        lambda g, grad: state.beta1 * g + (1 - state.beta1) * grad,
        state.momentum,
        avg_grads,
    )
    return state._replace(momentum=g)


def _compute_updates(params: optax.Updates, state: _State) -> optax.Updates:
    g_debias = tree_map(
        lambda g_i: g_i / (1 - state.beta1**state.current_step), state.momentum
    )
    return tree_map(
        lambda m, g, h: (g + state.weight_decay * m) / (h + state.weight_decay),
        params,
        g_debias,
        state.hessian,
    )


def _compute_variance(state: _State) -> optax.Updates:
    return tree_map(
        lambda h: rsqrt(state.ess * (h + state.weight_decay)), state.hessian
    )
