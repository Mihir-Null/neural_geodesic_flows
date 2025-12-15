"""
Toy data generator for 2D Schwarzschild radial infall geodesics.

Metric (t, r) slice:
    ds^2 = -f(r) dt^2 + f(r)^(-1) dr^2,   f(r) = 1 - 2M/r

We generate timelike radial infall geodesics parameterized by proper time τ.
Initial conditions are set by picking a starting radius r0 > 2M and a conserved
energy E >= sqrt(f(r0)) so that the normalization -f (dt/dτ)^2 + (1/f)(dr/dτ)^2 = -1 holds.
Then:
    dt/dτ = E / f(r0)
    dr/dτ = -sqrt(E^2 - f(r0))  (negative sign for infall)

Geodesic equations (non-zero Christoffels):
    Γ^t_{tr} = Γ^t_{rt} = f' / (2f)
    Γ^r_{tt} = f f' / 2
    Γ^r_{rr} = -f' / (2f)

ODE system on state (t, r, vt, vr):
    t'  = vt
    r'  = vr
    vt' = -(f'/f) * vt * vr
    vr' = -(f*f'/2) * vt^2 + (f'/(2f)) * vr^2

Saved datasets use the trajectory format expected by the NGF utilities:
  trajectories: shape (many, time_points, 4) with components (t, r, vt, vr)
  times: shape (time_points,) proper-time grid, shared across all trajectories
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from applications.configs import PATH_DATASETS


def f_and_fprime(r: np.ndarray, M: float) -> Tuple[np.ndarray, np.ndarray]:
    f = 1.0 - 2.0 * M / r
    fprime = 2.0 * M / (r ** 2)
    return f, fprime


def christoffels(r: float, M: float) -> Tuple[float, float, float]:
    f, fprime = f_and_fprime(r, M)
    gamma_t_tr = fprime / (2.0 * f)
    gamma_r_tt = 0.5 * f * fprime
    gamma_r_rr = -fprime / (2.0 * f)
    return gamma_t_tr, gamma_r_tt, gamma_r_rr


def geodesic_rhs(state: np.ndarray, M: float) -> np.ndarray:
    # state shape (..., 4)
    r = state[..., 1]
    vt = state[..., 2]
    vr = state[..., 3]
    gamma_t_tr, gamma_r_tt, gamma_r_rr = christoffels(r, M)

    dt_dtau = vt
    dr_dtau = vr
    dvt_dtau = -(2.0 * gamma_t_tr) * vt * vr
    dvr_dtau = -(gamma_r_tt) * vt ** 2 - (gamma_r_rr) * vr ** 2

    return np.stack([dt_dtau, dr_dtau, dvt_dtau, dvr_dtau], axis=-1)


def rk4_step(state: np.ndarray, h: float, M: float) -> np.ndarray:
    k1 = geodesic_rhs(state, M)
    k2 = geodesic_rhs(state + 0.5 * h * k1, M)
    k3 = geodesic_rhs(state + 0.5 * h * k2, M)
    k4 = geodesic_rhs(state + h * k3, M)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def sample_initial_conditions(
    num_traj: int,
    M: float,
    r_min: float,
    r_max: float,
    energy_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    r0 = rng.uniform(r_min, r_max, size=num_traj)
    energies = rng.uniform(energy_range[0], energy_range[1], size=num_traj)

    f0, _ = f_and_fprime(r0, M)
    # ensure energies are feasible: E >= sqrt(f0)
    energies = np.maximum(energies, np.sqrt(f0) + 1e-6)

    vt0 = energies / f0
    vr0 = -np.sqrt(np.maximum(energies ** 2 - f0, 0.0))

    t0 = np.zeros_like(r0)
    return np.stack([t0, r0, vt0, vr0], axis=-1)


def integrate_batch(
    init_states: np.ndarray, tau_final: float, num_steps: int, M: float
) -> np.ndarray:
    h = tau_final / num_steps
    batch_size = init_states.shape[0]
    traj = np.zeros((batch_size, num_steps + 1, init_states.shape[1]), dtype=np.float64)
    traj[:, 0, :] = init_states
    state = init_states.copy()
    for i in range(num_steps):
        state = rk4_step(state, h, M)
        traj[:, i + 1, :] = state
    return traj


def generate_schwarzschild_dataset(
    num_traj: int = 512,
    num_steps: int = 64,
    tau_final: float = 2.0,
    M: float = 1.0,
    r_min: float = 6.0,
    r_max: float = 8.0,
    energy_range: Tuple[float, float] = (0.98, 1.05),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    init_states = sample_initial_conditions(
        num_traj=num_traj,
        M=M,
        r_min=r_min,
        r_max=r_max,
        energy_range=energy_range,
        rng=rng,
    )
    trajectories = integrate_batch(
        init_states=init_states, tau_final=tau_final, num_steps=num_steps, M=M
    )
    times = np.linspace(0.0, tau_final, num_steps + 1, dtype=np.float64)
    times = np.broadcast_to(times, (num_traj, times.shape[0]))
    return trajectories, times


def save_dataset(
    name: str, trajectories: np.ndarray, times: np.ndarray, base_path: Path = PATH_DATASETS
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    np.savez(base_path / f"{name}.npz", trajectories=trajectories, times=times)
    print(f"Saved {name}.npz to {base_path} with shape {trajectories.shape}")


def main():
    train_traj, train_times = generate_schwarzschild_dataset(
        num_traj=512, num_steps=64, tau_final=8.0, seed=0
    )
    test_traj, test_times = generate_schwarzschild_dataset(
        num_traj=128, num_steps=64, tau_final=8.0, seed=1
    )

    save_dataset("schwarzschild_radial_train", train_traj, train_times)
    save_dataset("schwarzschild_radial_test", test_traj, test_times)


if __name__ == "__main__":
    main()
