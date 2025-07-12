import warnings

import matplotlib.pyplot as plt
import numpy as np

from bpm.mode_solver import slab_mode_source


def test_slab_mode_plot():
    # Define transverse coordinate and waveguide parameters
    x = np.linspace(-10, 10, 1000)  # Adjust range/resolution as needed
    w = 5.0  # waveguide width in microns
    n_WG = 1.1  # core refractive index
    n0 = 1.0  # cladding refractive index
    wavelength = 0.532  # in microns

    # Try to compute up to 5 modes
    num_modes = 5
    mode_fields = []
    mode_indices = []

    for m in range(num_modes):
        try:
            E_mode = slab_mode_source(x, w, n_WG, n0, wavelength, ind_m=m)
            mode_fields.append(E_mode)
            mode_indices.append(m)
        except Exception as err:
            warnings.warn(
                f"Mode {m} not found: {err}. Stopping mode search.", stacklevel=2
            )
            break

    # Plot the real part of each mode with an offset for clarity
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8, 6))
    offset = 1.0  # Vertical offset between mode plots
    for i, E_mode in enumerate(mode_fields):
        ax.plot(x, np.real(E_mode) + i * offset, label=f"Mode {mode_indices[i]}")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Field amplitude (Real part)")
    ax.set_title("Slab Waveguide TE Modes")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig("test_mode_solver.png")
    # plt.show()

    # Optionally convert the matplotlib figure to a Plotly figure for interactive viewing
    # plotly_fig = tls.mpl_to_plotly(fig)
    # plotly_fig.update_layout(legend=dict(font=dict(size=12)))
    # pio.show(plotly_fig)


if __name__ == "__main__":
    test_slab_mode_plot()
