import os
import glob
import subprocess
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAÇÕES DA ARQUITETURA
# ==============================================================================
PROJECT_DIR = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann"
BUILD_DIR = os.path.join(PROJECT_DIR, "cmake-build-release")
EXE_CMD = os.path.join(BUILD_DIR, "LatticeBoltzmann.exe")

TAU = 1.0
NU = (TAU - 0.5) / 3.0

# ==============================================================================
# MATRIZ DE TESTES PARAMÉTRICOS (K_imposto, U_inlet)
# ==============================================================================
TEST_MATRIX = [
    (1.0,  0.001),
    (2.0,  0.002),
    (5.0,  0.005),
    (10.0, 0.005),
    (20.0, 0.010),
    (50.0, 0.010)
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

def get_latest_vtk():
    result_dirs = glob.glob(os.path.join(BUILD_DIR, "results_*"))
    if not result_dirs: return None
    latest_dir = max(result_dirs, key=os.path.getmtime)
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))
    return max(vtk_files, key=os.path.getmtime) if vtk_files else None

def extract_permeability(vtk_file):
    mesh = pv.read(vtk_file)
    NX, NY, _ = mesh.dimensions

    ux_2d = mesh.point_data['velocity'][:, 0].reshape((NY, NX))
    rho_2d = mesh.point_data['rho'].reshape((NY, NX))
    p_2d = rho_2d / 3.0

    y_center = NY // 2
    bulk_start = int(0.25 * NX)
    bulk_end = int(0.75 * NX)

    x_bulk = np.arange(NX)[bulk_start:bulk_end]
    p_bulk = p_2d[y_center, bulk_start:bulk_end]
    u_bulk = ux_2d[y_center, bulk_start:bulk_end]

    coeffs = np.polyfit(x_bulk, p_bulk, 1)
    dp_dx = coeffs[0]

    u_avg_bulk = np.mean(u_bulk)
    rho_avg_bulk = np.mean(rho_2d[y_center, bulk_start:bulk_end])

    K_rec = (NU * rho_avg_bulk * u_avg_bulk) / (-dp_dx)
    return K_rec

def run_pipeline():
    results_k_imposto = []
    results_k_recuperado = []
    results_erros = []

    print("==================================================")
    print("INICIANDO VARREDURA PARAMÉTRICA (GPU)")
    print("==================================================")

    for k_imp, u_imp in TEST_MATRIX:
        print(f"-> Executando LBM | K_0 = {k_imp:5.1f} | U = {u_imp:.4f}")

        # Injeção Direta por Argumentos
        run_cmd = [EXE_CMD, str(k_imp), str(u_imp)]

        try:
            # Executa o LBM ocultando a saída do console para não poluir o terminal Python
            subprocess.run(run_cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("   [ERRO FATAL] Divergência numérica LBM.")
            return

        vtk_path = get_latest_vtk()
        k_rec = extract_permeability(vtk_path)
        erro_rel = abs(k_rec - k_imp) / k_imp * 100.0

        results_k_imposto.append(k_imp)
        results_k_recuperado.append(k_rec)
        results_erros.append(erro_rel)

        print(f"   K recuperado: {k_rec:7.4f} | Erro Relativo: {erro_rel:5.3f} %")

    print("\nProcessando análises de erro...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.plot(results_k_imposto, results_k_imposto, 'k--', linewidth=1.5, label='Ideal Parity')
    ax1.plot(results_k_imposto, results_k_recuperado, 'ko', markeredgecolor='black', markerfacecolor='none',
             markersize=7, label='Numerical LBM')
    ax1.set_xlabel(r'Imposed Permeability, $K_{imposed}$ [lu]')
    ax1.set_ylabel(r'Recovered Permeability, $K_{recovered}$ [lu]')
    ax1.set_title('Darcy Permeability Recovery', fontsize=12)
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(results_k_imposto, results_erros, 'ks-', linewidth=1.5, markersize=6, markerfacecolor='black')
    ax2.set_xlabel(r'Imposed Permeability, $K_{imposed}$ [lu]')
    ax2.set_ylabel(r'Relative Error [%]')
    ax2.set_title('Computational Error Analysis', fontsize=12)
    ax2.set_ylim(bottom=0.0)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_DIR, "permeability_error_analysis.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {plot_path}")
    plt.show()

if __name__ == '__main__':
    run_pipeline()