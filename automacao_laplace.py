import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# Força o estilo acadêmico rigoroso de plotagem
plt.style.use('default')
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

BASE_DIR = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"

def extract_parameters_from_log(log_path):
    params = {}
    if not os.path.exists(log_path):
        return params
    with open(log_path, 'r') as file:
        for line in file:
            if "=" in line:
                parts = line.split("=")
                key = parts[0].strip()
                val_str = parts[1].strip()
                try:
                    params[key] = float(val_str)
                except ValueError:
                    pass
    return params

def process_all_results():
    results_dirs = glob.glob(os.path.join(BASE_DIR, "results_*"))

    curvaturas = []
    delta_ps = []

    ref_log = os.path.join(results_dirs[-1], "config_log.txt") if results_dirs else None
    params = extract_parameters_from_log(ref_log)

    if "SIGMA" not in params:
        print("Error: config_log.txt not found or missing SIGMA parameter.")
        return

    sigma_theo = params["SIGMA"]
    w_int = params["INTERFACE_WIDTH"]

    beta = (3.0 * sigma_theo) / (8.0 * w_int)
    kappa = (3.0 * sigma_theo * w_int) / 4.0

    print(f"Dynamic Parameters: SIGMA={sigma_theo:.2e}, W={w_int:.2f}, BETA={beta:.6e}, KAPPA={kappa:.6e}")

    for folder in results_dirs:
        config_path = os.path.join(folder, "config.txt")
        if not os.path.exists(config_path):
            continue

        vtk_files = glob.glob(os.path.join(folder, "data_*.vtk"))
        if not vtk_files:
            continue
        latest_vtk = max(vtk_files, key=os.path.getmtime)

        mesh = pv.read(latest_vtk)
        NX, NY, _ = mesh.dimensions

        try:
            rho_field = mesh.point_data['rho'].reshape((NY, NX))
            phi_field = mesh.point_data['phi'].reshape((NY, NX))
        except KeyError:
            continue

        area_fase = np.sum(phi_field > 0.0)
        if area_fase == 0:
            continue
        r_num = np.sqrt(area_fase / np.pi)

        # Filtro de microgotas opcional para evitar o erro de alavancagem em R pequenos (R < 20)
        # if r_num < 20.0:
        #     continue

        lap_phi = laplace(phi_field, mode='wrap')
        p_mech = rho_field / 3.0

        free_energy_bulk = beta * (phi_field**2 - 1.0) * (3.0 * phi_field**2 + 1.0)
        p_th_field = p_mech + free_energy_bulk - kappa * phi_field * lap_phi

        mask_inside = phi_field > 0.95
        mask_outside = phi_field < -0.95

        p_in = np.mean(p_th_field[mask_inside])
        p_out = np.mean(p_th_field[mask_outside])

        dp = p_in - p_out
        curvatura = 1.0 / r_num

        curvaturas.append(curvatura)
        delta_ps.append(dp)

        print(f"[{os.path.basename(folder)}] R={r_num:.4f} | Curv={curvatura:.5f} | DP={dp:.6e} | (P_in={p_in:.6e}, P_out={p_out:.6e})")

    if not curvaturas:
        print("No valid data to plot.")
        return

    coeffs = np.polyfit(curvaturas, delta_ps, 1)
    sigma_num = coeffs[0]
    p_spurious = coeffs[1]
    erro_sigma = abs(sigma_num - sigma_theo) / sigma_theo * 100.0

    print("\n" + "="*50)
    print("MACROSCOPIC SURFACE TENSION ANALYSIS")
    print("="*50)
    print(f"Theoretical Sigma (C-H Integral) : {sigma_theo:.6e}")
    print(f"Numerical Sigma (Regression)     : {sigma_num:.6e}")
    print(f"Relative Error                   : {erro_sigma:.2f} %")
    print(f"Residual Spurious Pressure       : {p_spurious:.6e}")

    plt.figure(figsize=(7, 5))

    x_plot = np.linspace(0, max(curvaturas)*1.1, 100)
    plt.plot(x_plot, sigma_theo * x_plot, 'k--', label=fr'Theoretical ($\sigma_{{theo}}={sigma_theo:.4e}$)')
    plt.plot(x_plot, sigma_num * x_plot + p_spurious, 'r-', label=fr'Numerical line ($\sigma_{{num}}={sigma_num:.4e}$)')
    plt.plot(curvaturas, delta_ps, 'ko', markersize=7, markerfacecolor='none', markeredgewidth=1.5, label='Simulation results')

    plt.title('Validation of the Laplace-Young Law', fontsize=12)
    plt.xlabel(r'Interface Curvature, $\frac{1}{R}$ [lu$^{-1}$]')
    plt.ylabel(r'Thermodynamic Pressure Jump, $\Delta P_{th}$ [lu]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("laplace_regression_acumulada.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    process_all_results()