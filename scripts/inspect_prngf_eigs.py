"""
Inspect metric eigenvalues of saved RNGF/PRNGF models to verify definiteness.
RNGF models should be SPD (all positive). PRNGF should show a negative eigenvalue.
"""

import sys
from pathlib import Path

# ensure repo root on path so "applications" is importable when run from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from applications.utils import load_model
from core.template_psi_phi_g_functions_neural_networks import (
    NN_Jacobian_split_diffeomorphism,
    NN_pseudo_metric_fixed_signature,
    NN_metric_regularized,
)


def metric_eigs(model):
    g = model.g(jnp.array([0.1, 0.2]))
    g = 0.5 * (g + g.T)
    return jnp.linalg.eigvalsh(g)


def main():
    pairs = [
        ("minkowski_rngf", NN_metric_regularized),
        ("minkowski_prngf", NN_pseudo_metric_fixed_signature),
        ("ads2_rngf", NN_metric_regularized),
        ("ads2_prngf", NN_pseudo_metric_fixed_signature),
        ("schwarzschild_rngf", NN_metric_regularized),
        ("schwarzschild_prngf", NN_pseudo_metric_fixed_signature),
    ]

    for name, g_init in pairs:
        try:
            model = load_model(
                name,
                psi_initializer=NN_Jacobian_split_diffeomorphism,
                phi_initializer=NN_Jacobian_split_diffeomorphism,
                g_initializer=g_init,
            )
            evals = metric_eigs(model)
            print(f"{name}: {evals}")
        except Exception as e:
            print(f"{name} failed: {e}")


if __name__ == "__main__":
    main()
