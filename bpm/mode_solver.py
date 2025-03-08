import numpy as np
import warnings

def slab_mode_source(x, w, n_WG, n0, wavelength, ind_m=0, x0=0):
    """
    Returns the normalized TE mode profile for a symmetric slab waveguide with a lateral shift x0.
    """
    k0 = 2 * np.pi / wavelength

    def f_even(beta):
        if beta < n0*k0 or beta > n_WG*k0:
            return None
        inside = n_WG**2 * k0**2 - beta**2
        outside = beta**2 - n0**2 * k0**2
        if inside <= 0 or outside <= 0:
            return None
        kx = np.sqrt(inside)
        kappa = np.sqrt(outside)
        return kx * np.tan(kx * w / 2) - kappa

    def f_odd(beta):
        if beta < n0*k0 or beta > n_WG*k0:
            return None
        inside = n_WG**2 * k0**2 - beta**2
        outside = beta**2 - n0**2 * k0**2
        if inside <= 0 or outside <= 0:
            return None
        kx = np.sqrt(inside)
        kappa = np.sqrt(outside)
        sin_term = np.sin(kx * w / 2)
        if abs(sin_term) < 1e-12:
            return None
        return - kx * (np.cos(kx * w / 2) / sin_term) - kappa

    def valid_even(beta):
        inside = n_WG**2 * k0**2 - beta**2
        if inside <= 0:
            return False
        kx = np.sqrt(inside)
        theta = kx * w / 2
        m = int(np.floor(2 * theta / np.pi))
        if m % 2 == 0:
            if m == 0 and theta > (np.pi/2 - 0.1):
                return False
            return True
        return False

    def valid_odd(beta):
        inside = n_WG**2 * k0**2 - beta**2
        if inside <= 0:
            return False
        kx = np.sqrt(inside)
        theta = kx * w / 2
        m = int(np.floor(2 * theta / np.pi))
        return (m % 2 == 1)

    N = 2000
    beta_scan = np.linspace(n0*k0, n_WG*k0, N)
    even_intervals = []
    odd_intervals = []
    f_even_vals = [f_even(b) for b in beta_scan]
    f_odd_vals = [f_odd(b) for b in beta_scan]
    for i in range(N-1):
        if (f_even_vals[i] is not None) and (f_even_vals[i+1] is not None):
            if f_even_vals[i] * f_even_vals[i+1] < 0:
                even_intervals.append((beta_scan[i], beta_scan[i+1]))
        if (f_odd_vals[i] is not None) and (f_odd_vals[i+1] is not None):
            if f_odd_vals[i] * f_odd_vals[i+1] < 0:
                odd_intervals.append((beta_scan[i], beta_scan[i+1]))
                
    def refine_root(f, b_left, b_right):
        for _ in range(50):
            b_mid = 0.5*(b_left+b_right)
            val_mid = f(b_mid)
            if val_mid is None:
                b_right = b_mid
                continue
            if abs(val_mid) < 1e-9:
                return b_mid
            val_left = f(b_left)
            if val_left is None or val_left*val_mid > 0:
                b_left = b_mid
            else:
                b_right = b_mid
        return b_mid

    even_roots = []
    for (b_left, b_right) in even_intervals:
        root = refine_root(f_even, b_left, b_right)
        if valid_even(root):
            even_roots.append(root)
    odd_roots = []
    for (b_left, b_right) in odd_intervals:
        root = refine_root(f_odd, b_left, b_right)
        if valid_odd(root):
            odd_roots.append(root)

    modes = [("even", r) for r in even_roots] + [("odd", r) for r in odd_roots]
    modes_sorted = sorted(modes, key=lambda tup: tup[1], reverse=True)
    if len(modes_sorted) == 0:
        raise ValueError("No guided slab modes found in [n0*k0, n_WG*k0].")
    if ind_m >= len(modes_sorted):
        warnings.warn(
            f"Requested mode index {ind_m} >= found modes ({len(modes_sorted)}). Using highest mode index {len(modes_sorted)-1}.",
            UserWarning
        )
        ind_m = len(modes_sorted) - 1

    parity, beta_chosen = modes_sorted[ind_m]
    inside = n_WG**2 * k0**2 - beta_chosen**2
    outside = beta_chosen**2 - n0**2 * k0**2
    kx = np.sqrt(inside)
    kappa = np.sqrt(outside)

    E = np.zeros_like(x, dtype=np.complex128)
    if parity == "even":
        for i, xi in enumerate(x):
            xp = xi - x0
            if abs(xp) <= w/2:
                E[i] = np.cos(kx * xp)
            else:
                E[i] = np.cos(kx * (w/2)) * np.exp(-kappa * (abs(xp)-w/2))
    else:
        for i, xi in enumerate(x):
            xp = xi - x0
            if abs(xp) <= w/2:
                E[i] = np.sin(kx * xp)
            else:
                E[i] = np.sign(xp) * np.sin(kx * (w/2)) * np.exp(-kappa * (abs(xp)-w/2))
    norm = np.sqrt(np.trapz(np.abs(E)**2, x))
    E /= norm
    return E
