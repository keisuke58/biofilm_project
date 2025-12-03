# main_simulation.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import CONFIG, get_theta_true
from src.solver_newton import BiofilmNewtonSolver

def _save_timeseries(outputs: dict, output_dir: Path) -> None:
    """Save combined time-series (CSV/NPZ) for a single model run."""

    model_name = outputs["model_name"]
    t = outputs["t"]
    g = outputs["g"]

    np.savez(output_dir / f"case2_{model_name}_timeseries.npz", t=t, g=g)

    header_columns = [
        "time",
        "phi1",
        "phi2",
        "phi3",
        "phi4",
        "phi0",
        "psi1",
        "psi2",
        "psi3",
        "psi4",
        "gamma",
    ]
    np.savetxt(
        output_dir / f"case2_{model_name}_timeseries.csv",
        np.column_stack([t, g]),
        delimiter=",",
        header=",".join(header_columns),
        comments="",
    )


def _plot_timeseries(outputs: dict, output_dir: Path) -> None:
    """Plot the phi time series for a model run as a PNG artifact."""

    model_name = outputs["model_name"]
    t = outputs["t"]
    g = outputs["g"]
    phi = g[:, :4]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for idx in range(phi.shape[1]):
        ax.plot(t, phi[:, idx], label=fr"$\phi_{idx + 1}(t)$")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\phi_\ell(t)$")
    ax.set_title(f"Case II {model_name} forward simulation")
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output_dir / f"case2_{model_name}_timeseries.png", dpi=200)
    plt.close(fig)


def _run_model(model_name: str, phi_init_key: str) -> dict:
    config = CONFIG
    theta_true = get_theta_true()

    solver = BiofilmNewtonSolver(
        phi_init=config[phi_init_key], use_numba=True, **config[model_name]
    )
    t, g = solver.run_deterministic(theta_true, show_progress=True)

    print(f"{model_name}: t.shape={t.shape}, g.shape={g.shape}")
    return {"model_name": model_name, "t": t, "g": g}


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    runs = [
        _run_model("M1", "phi_init_M1"),
        _run_model("M2", "phi_init_M2"),
        _run_model("M3", "phi_init_M3"),
    ]

    for run_output in runs:
        _save_timeseries(run_output, output_dir)
        _plot_timeseries(run_output, output_dir)

    # Also save a combined NPZ for convenience/compatibility with earlier workflows.
    combined_payload = {}
    for run_output in runs:
        model_name = run_output["model_name"]
        combined_payload[f"{model_name}_t"] = run_output["t"]
        combined_payload[f"{model_name}_g"] = run_output["g"]

    np.savez(output_dir / "forward_simulation_timeseries.npz", **combined_payload)

    print("Simulation finished.")
    print(
        f"Saved results to {output_dir} (case2_M1/M2/M3 timeseries, PNGs, and combined NPZ)"
    )


if __name__ == "__main__":
    main()
