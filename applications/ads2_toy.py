"""
Toy data generator for AdS_2 geodesics via ambient embedding in R^{2,1}.

Ambient Minkowski metric: diag(-1, -1, +1)
Manifold: AdS_2 = { X in R^{2,1} | <X, X> = -1 }

Timelike geodesics with tangent V0 (⟨X0, V0⟩ = 0, ⟨V0, V0⟩ = -1):
    X(tau) = cos(tau) X0 + sin(tau) V0
    X'(tau) = -sin(tau) X0 + cos(tau) V0

We generate bounded initial points and timelike tangent vectors, then evaluate
the closed-form geodesics on a uniform tau grid.

Saved datasets use the trajectory format expected by the NGF utilities:
  trajectories: shape (many, time_points, 6) with components (X0,X1,X2,V0,V1,V2)
  times: shape (time_points,) proper-time grid, broadcast across the batch
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from applications.configs import PATH_DATASETS


def minkowski_inner(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # diag(-1, -1, +1)
    return -u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def sample_point(rng: np.random.Generator,
                 x1_range: Tuple[float, float],
                 x2_range: Tuple[float, float]) -> np.ndarray:
    x1 = rng.uniform(*x1_range)
    x2 = rng.uniform(*x2_range)
    # enforce <X,X> = -1 => x0^2 = x2^2 - x1^2 + 1
    rad = x2 * x2 - x1 * x1 + 1.0
    x0 = np.sqrt(rad)
    return np.array([x0, x1, x2], dtype=np.float64)


def sample_timelike_tangent(rng: np.random.Generator, X: np.ndarray,
                            max_tries: int = 100) -> np.ndarray:
    for _ in range(max_tries):
        W = rng.normal(size=3)
        # project to tangent: V = W + <X,W> X  (since <X,X>=-1)
        inner_XW = minkowski_inner(X, W)
        V = W + inner_XW * X
        norm = minkowski_inner(V, V)
        if norm < 0:
            scale = 1.0 / np.sqrt(-norm)  # make norm = -1
            return V * scale
    raise RuntimeError("Failed to sample timelike tangent vector")


def generate_ads2_dataset(
    num_traj: int = 512,
    num_steps: int = 64,
    tau_final: float = np.pi,  # covers a full timelike oscillation
    x1_range: Tuple[float, float] = (-0.8, 0.8),
    x2_range: Tuple[float, float] = (1.0, 1.6),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, tau_final, num_steps + 1, dtype=np.float64)
    times_b = np.broadcast_to(times, (num_traj, times.shape[0]))

    traj = np.zeros((num_traj, times.shape[0], 6), dtype=np.float64)

    for i in range(num_traj):
        X0 = sample_point(rng, x1_range, x2_range)
        V0 = sample_timelike_tangent(rng, X0)

        cos_t = np.cos(times)
        sin_t = np.sin(times)

        X_tau = cos_t[:, None] * X0[None, :] + sin_t[:, None] * V0[None, :]
        V_tau = -sin_t[:, None] * X0[None, :] + cos_t[:, None] * V0[None, :]

        traj[i, :, :3] = X_tau
        traj[i, :, 3:] = V_tau

    return traj, times_b


def save_dataset(
    name: str, trajectories: np.ndarray, times: np.ndarray, base_path: Path = PATH_DATASETS
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    np.savez(base_path / f"{name}.npz", trajectories=trajectories, times=times)
    print(f"Saved {name}.npz to {base_path} with shape {trajectories.shape}")


def main():
    train_traj, train_times = generate_ads2_dataset(
        num_traj=512, num_steps=64, tau_final=np.pi, seed=0
    )
    test_traj, test_times = generate_ads2_dataset(
        num_traj=128, num_steps=64, tau_final=np.pi, seed=1
    )

    save_dataset("ads2_train", train_traj, train_times)
    save_dataset("ads2_test", test_traj, test_times)


if __name__ == "__main__":
    main()
