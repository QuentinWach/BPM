import tempfile
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from bpm.core import run_bpm
from bpm.mode_solver import slab_mode_source
from bpm.pml import generate_sigma_x
from bpm.refractive_index import generate_waveguide_n_r2


def run_waveguide(
    w: float, l: float, L: float, n_WG: float, wavelength: float, ind_m: int
) -> tuple[Any, str]:
    # Simulation setup
    domain_size = 50.0
    z_total = 500.0
    Nx, Nz = 256, 2000
    x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
    z = np.linspace(0, z_total, Nz)
    n0 = 1.0

    # Refractive index map
    n_r2 = generate_waveguide_n_r2(x, z, l, L, w, n_WG, n0)

    # Mode source
    E0 = slab_mode_source(x, w, n_WG, n0, wavelength, ind_m, x0=0)
    E = np.zeros((Nx, Nz), dtype=np.complex128)
    E[:, 0] = E0

    # PML and BPM propagation
    dx = domain_size / Nx
    dz = z[1] - z[0]
    sigma_x = generate_sigma_x(x, dx, wavelength, domain_size)
    E_out = run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.abs(E_out) ** 2,
        extent=(x[0], x[-1], z[0], z[-1]),
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title("Waveguide BPM Propagation")
    fig.colorbar(im, ax=ax, label="Intensity")

    # Save data for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
        np.savez(tmp_file.name, E_out=E_out, x=x, z=z)

    return fig, tmp_file.name


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Waveguide BPM Simulation")

    with gr.Row():
        with gr.Column(scale=1):
            w_slider = gr.Slider(
                0.1, 5.0, value=1.0, step=0.1, label="Waveguide width w (µm)"
            )
            l_slider = gr.Slider(
                0.0, 10.0, value=5.0, step=0.1, label="Lateral offset l (µm)"
            )
            L_slider = gr.Slider(
                50.0, 500.0, value=200.0, step=10.0, label="S-bend length L (µm)"
            )
            n_WG_slider = gr.Slider(
                1.0, 2.0, value=1.1, step=0.01, label="Core refractive index n_WG"
            )
            wavelength_slider = gr.Slider(
                0.4, 1.6, value=0.532, step=0.01, label="Wavelength λ (µm)"
            )
            ind_m_slider = gr.Slider(0, 4, value=0, step=1, label="Mode index ind_m")

            run_button = gr.Button("Run BPM")
            download_button = gr.DownloadButton(label="Download data")

        with gr.Column(scale=2):
            plot_output = gr.Plot()

    inputs = [
        w_slider,
        l_slider,
        L_slider,
        n_WG_slider,
        wavelength_slider,
        ind_m_slider,
    ]

    # Connect run button
    run_button.click(
        fn=run_waveguide, inputs=inputs, outputs=[plot_output, download_button]
    )

    # Auto-update on parameter change
    for inp in inputs:
        inp.change(
            fn=run_waveguide, inputs=inputs, outputs=[plot_output, download_button]
        )


def main() -> None:
    """Main entry point for the BPM GUI application."""
    demo.launch()


if __name__ == "__main__":
    main()
