"""
Compare pseudo-Riemannian (PRNGF) and Riemannian (RNGF) models across all
available trajectory datasets and report their trajectory losses side by side.

This adapts the AdS2 demo idea to a multi-dataset setting: it loads the trained
models, evaluates the same test split, and prints the losses so you can see
where the Lorentzian-capable model wins.

Expected pre-trained models (saved under data/models):
  - schwarzschild_prngf / schwarzschild_rngf
  - minkowski_prngf    / minkowski_rngf
  - ads2_prngf         / ads2_rngf
  - sphere_prngf       / sphere_rngf
  - two_body_prngf     / two_body_rngf
If a model file is missing, that entry is skipped.
"""

from pathlib import Path
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import equinox as eqx

from core.losses import trajectory_loss
from core.template_psi_phi_g_functions_neural_networks import (
    NN_Jacobian_split_diffeomorphism,
    NN_metric_regularized,
    NN_pseudo_metric_fixed_signature,
)
from applications.utils import load_model, load_dataset
from applications.configs import PATH_MODELS


DatasetCfg = Dict[str, str]


def model_exists(name: str) -> bool:
    return (PATH_MODELS / f"{name}.eqx").exists()


def eval_loss(
    model,
    trajectories: jnp.ndarray,
    times: jnp.ndarray,
    metric_reg_weight: float,
    metric_logabsdet_floor: float = -12.0,
) -> float:
    fn = eqx.filter_jit(
        lambda m, tr, ti: trajectory_loss(
            m,
            tr,
            ti,
            metric_reg_weight=metric_reg_weight,
            metric_logabsdet_floor=metric_logabsdet_floor,
        )
    )
    loss = fn(model, trajectories, times)
    return float(loss)


def load_pair(
    rngf_name: str, prngf_name: str, dim_dataspace: int
):
    # both use the same psi/phi initializer; g differs
    psi_init = NN_Jacobian_split_diffeomorphism
    phi_init = NN_Jacobian_split_diffeomorphism

    rngf_model = load_model(
        rngf_name, psi_initializer=psi_init, phi_initializer=phi_init, g_initializer=NN_metric_regularized
    )
    prngf_model = load_model(
        prngf_name, psi_initializer=psi_init, phi_initializer=phi_init, g_initializer=NN_pseudo_metric_fixed_signature
    )

    if rngf_model is None or prngf_model is None:
        raise ValueError(f"Failed to load models {rngf_name} / {prngf_name}")
    # sanity check on dataspace dimension to avoid silent mismatches
    assert rngf_model.dim_dataspace == dim_dataspace
    assert prngf_model.dim_dataspace == dim_dataspace
    return rngf_model, prngf_model


def main():
    datasets: List[Dict] = [
        {
            "name": "minkowski_flat",
            "dim_dataspace": 4,
            "rngf": "minkowski_rngf",
            "prngf": "minkowski_prngf",
            "eval_size": None,
        },
        {
            "name": "ads2",
            "dim_dataspace": 6,
            "rngf": "ads2_rngf",
            "prngf": "ads2_prngf",
            "eval_size": None,
        },
        {
            "name": "sphere_trajectories",
            "dim_dataspace": 6,
            "rngf": "sphere_rngf",
            "prngf": "sphere_prngf",
            "eval_size": 512,
        },
        {
            "name": "two-body-problem_trajectories",
            "dim_dataspace": 8,
            "rngf": "two_body_rngf",
            "prngf": "two_body_prngf",
            "eval_size": 512,
        },
    ]

    rows = []

    for cfg in datasets:
        ds_test = f"{cfg['name']}_test"
        if not (model_exists(cfg["rngf"]) and model_exists(cfg["prngf"])):
            print(f"Skipping {cfg['name']}: missing model file(s).")
            continue

        print(f"Evaluating {cfg['name']} ...")
        data, mode = load_dataset(
            name=ds_test,
            size=cfg.get("eval_size"),
            random_selection=True,
            key=jax.random.PRNGKey(0),
        )
        if mode != "trajectory":
            print(f"  Dataset {ds_test} is not a trajectory dataset, skipping.")
            continue
        trajectories, times = data

        rngf_model, prngf_model = load_pair(cfg["rngf"], cfg["prngf"], cfg["dim_dataspace"])

        loss_rngf = eval_loss(rngf_model, trajectories, times, metric_reg_weight=0.0)
        loss_prngf = eval_loss(prngf_model, trajectories, times, metric_reg_weight=1e-2)

        rows.append((cfg["name"], loss_rngf, loss_prngf))
        print(f"  RNGF loss: {loss_rngf:.6e} | PRNGF loss: {loss_prngf:.6e}")

    print("\n=== Summary (test trajectory loss) ===")
    for name, l_r, l_pr in rows:
        rel = (l_r - l_pr) / max(abs(l_r), 1e-12)
        print(f"{name:32s}  RNGF: {l_r:.6e}  PRNGF: {l_pr:.6e}  (rel diff: {rel:+.2%})")


if __name__ == "__main__":
    main()
