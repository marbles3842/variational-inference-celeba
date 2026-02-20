import jax
import optax
from clu import metrics
from flax.struct import dataclass
from flax.training.train_state import TrainState


@dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    hess_mean: metrics.Average.from_output("hess_mean")
    hess_min: metrics.Average.from_output("hess_min")


class TrainState(TrainState):
    metrics: Metrics
    ema_params: dict
    ema_step_size: float

    def apply_ema(self):
        new_ema_params = optax.incremental_update(
            new_tensors=self.params,
            old_tensors=self.ema_params,
            step_size=self.ema_step_size,
        )
        return self.replace(ema_params=new_ema_params)


def create_train_state(
    model, rng: jax.random.PRNGKey, x0: jax.Array, t0: jax.Array, optimizer,
    ema_step_size: float = 0.001,
):
    variables = model.init(rng, x0, t0)
    params = variables["params"]
    ema_params = jax.tree.map(jax.numpy.copy, params)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=Metrics.empty(),
        ema_params=ema_params,
        ema_step_size=ema_step_size,
    )
