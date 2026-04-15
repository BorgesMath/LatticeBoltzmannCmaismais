import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
TAU = 1.0
NU = (TAU - 0.5) / 3.0

def process_all_results():
    results_dirs = glob.glob(os.path.join(BASE_DIR, "results_*"))

    k_imposed_list = []
    k_recovered_list = []

    print(f"Encontradas {len(results_dirs)} pastas de resultados.")

    for folder in results_dirs:
        config_path = os.path.join(folder, "config.txt")
        if not os.path.exists(config_path):
            continue

        with open(config_path, 'r') as f:
            lines = f.readlines()
            k_imp = float(lines[0].strip())
            u_imp = float(lines[1].strip())

        vtk_files = glob.glob(os.path.join(folder, "data_*.vtk"))
        if not vtk_files:
            continue
        latest_vtk = max(vtk_files, key=os.path.getmtime)

        mesh = pv.read(latest_vtk)
        NX, NY, _ = mesh.dimensions

        # Blindagem contra arquivos corrompidos por divergência (NaN)
        try:
            ux_2d = mesh.point_data['velocity'][:, 0].reshape((NY, NX))
            rho_2d = mesh.point_data['rho'].reshape((NY, NX))
        except KeyError:
            print(f"Pasta {os.path.basename(folder)}: Divergência Numérica (NaN) detectada. Ignorando...")
            continue

        p_2d = rho_2d / 3.0

        y_center = NY // 2
        bulk_start, bulk_end = int(0.25 * NX), int(0.75 * NX)
        x_bulk = np.arange(NX)[bulk_start:bulk_end]
        p_bulk = p_2d[y_center, bulk_start:bulk_end]
        u_avg_bulk = np.mean(ux_2d[y_center, bulk_start:bulk_end])
        rho_avg_bulk = np.mean(rho_2d[y_center, bulk_start:bulk_end])

        coeffs = np.polyfit(x_bulk, p_bulk, 1)
        dp_dx = coeffs[0]

        if abs(dp_dx) > 1e-15:
            k_rec = (NU * rho_avg_bulk * u_avg_bulk) / (-dp_dx)
            k_imposed_list.append(k_imp)
            k_recovered_list.append(k_rec)
            print(f"Pasta {os.path.basename(folder)}: Imposto={k_imp:.2f}, Recuperado={k_rec:.4f}")

    if not k_imposed_list:
        print("Nenhum dado válido para plotar.")
        return

    idx_sort = np.argsort(k_imposed_list)
    k_imp_arr = np.array(k_imposed_list)[idx_sort]
    k_rec_arr = np.array(k_recovered_list)[idx_sort]
    errors = np.abs(k_imp_arr - k_rec_arr) / k_imp_arr * 100

    plt.style.use('default')
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_imp_arr, k_imp_arr, 'k--', label='Ideal (Parity)')
    ax1.scatter(k_imp_arr, k_rec_arr, color='black', facecolors='none', edgecolors='k', label='LBM Recovery')
    ax1.set_xlabel(r'Imposed Permeability $K_0$ [lu]')
    ax1.set_ylabel(r'Recovered Permeability $K_{num}$ [lu]')
    ax1.legend(frameon=False)
    ax1.set_title('Darcy Validation: K Recovery')

    ax2.plot(k_imp_arr, errors, 'ks-', markersize=4)
    ax2.set_xlabel(r'Imposed Permeability $K_0$ [lu]')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Computational Error Analysis')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("validacao_darcy_acumulada.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    process_all_results()