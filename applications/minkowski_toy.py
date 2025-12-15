"""
Toy data generator for flat 2D Minkowski spacetime in bounded coordinates.

Metric: g = diag(-1, +1) on R^{1,1}
Timelike geodesics are straight lines:
    x^mu(tau) = x0^mu + v^mu * tau
with normalization -vt^2 + vx^2 = -1 (future-directed, vt > 0).

We sample bounded positions and spatial velocities, enforce the timelike
condition by setting vt = sqrt(1 + vx^2), and integrate over tau in [0, 1]
with a fixed number of steps.

Saved datasets use the trajectory format expected by the NGF utilities:
  trajectories: shape (many, time_points, 4) with components (t, x, vt, vx)
  times: shape (time_points,) proper-time grid, broadcast across the batch
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from applications.configs import PATH_DATASETS


def sample_initial_conditions(
    num_traj: int,
    pos_range: Tuple[float, float],
    vx_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    t0 = rng.uniform(pos_range[0], pos_range[1], size=num_traj)
    x0 = rng.uniform(pos_range[0], pos_range[1], size=num_traj)
    vx = rng.uniform(vx_range[0], vx_range[1], size=num_traj)
    vt = np.sqrt(1.0 + vx ** 2)  # enforce timelike normalization, future-directed
    return np.stack([t0, x0, vt, vx], axis=-1)


def generate_minkowski_dataset(
    num_traj: int = 512,
    num_steps: int = 32,
    tau_final: float = 1.0,
    pos_range: Tuple[float, float] = (-1.0, 1.0),
    vx_range: Tuple[float, float] = (-0.5, 0.5),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    init_states = sample_initial_conditions(
        num_traj=num_traj, pos_range=pos_range, vx_range=vx_range, rng=rng
    )
    # straight-line integration
    times = np.linspace(0.0, tau_final, num_steps + 1, dtype=np.float64)
    traj = np.zeros((num_traj, times.shape[0], init_states.shape[1]), dtype=np.float64)
    vt = init_states[:, 2][:, None]
    vx = init_states[:, 3][:, None]
    traj[:, :, 0] = init_states[:, 0][:, None] + vt * times  # t(tau)
    traj[:, :, 1] = init_states[:, 1][:, None] + vx * times  # x(tau)
    traj[:, :, 2] = vt  # constant vt
    traj[:, :, 3] = vx  # constant vx
    times = np.broadcast_to(times, (num_traj, times.shape[0]))
    return traj, times


def save_dataset(
    name: str, trajectories: np.ndarray, times: np.ndarray, base_path: Path = PATH_DATASETS
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    np.savez(base_path / f"{name}.npz", trajectories=trajectories, times=times)
    print(f"Saved {name}.npz to {base_path} with shape {trajectories.shape}")


def main():
    train_traj, train_times = generate_minkowski_dataset(
        num_traj=512, num_steps=32, tau_final=1.0, seed=0
    )
    test_traj, test_times = generate_minkowski_dataset(
        num_traj=128, num_steps=32, tau_final=1.0, seed=1
    )

    save_dataset("minkowski_flat_train", train_traj, train_times)
    save_dataset("minkowski_flat_test", test_traj, test_times)


if __name__ == "__main__":
    main()
