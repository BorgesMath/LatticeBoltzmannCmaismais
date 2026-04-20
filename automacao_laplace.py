import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"

# Baseado nos parâmetros atuais do seu config.cuh

SIGMA_THEO = 0.001

def process_all_results():
    results_dirs = glob.glob(os.path.join(BASE_DIR, "results_*"))

    curvaturas = []
    delta_ps = []

    print(f"Encontradas {len(results_dirs)} pastas de resultados.")

    for folder in results_dirs:
        # Pula pastas que não têm o config de Laplace
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

        # Raio Efetivo (Área Numérica)
        area_fase = np.sum(phi_field > 0.0)
        if area_fase == 0:
            continue
        r_num = np.sqrt(area_fase / np.pi)

        # Pressão no Bulk
        centro_y, centro_x = NY // 2, NX // 2
        p_in = np.mean(rho_field[centro_y-2:centro_y+3, centro_x-2:centro_x+3]) / 3.0
        p_out = np.mean(rho_field[0:5, 0:5]) / 3.0

        dp = p_in - p_out
        curvatura = 1.0 / r_num

        curvaturas.append(curvatura)
        delta_ps.append(dp)

        print(f"[{os.path.basename(folder)}] R_num={r_num:.4f} | Curvatura={curvatura:.5f} | DP={dp:.6e}")

    if not curvaturas:
        print("Nenhum dado válido para plotar.")
        return

    # Regressão Linear: DP = sigma_num * (1/R) + Erro
    coeffs = np.polyfit(curvaturas, delta_ps, 1)
    sigma_num = coeffs[0]
    p_spurious = coeffs[1]
    erro_sigma = abs(sigma_num - SIGMA_THEO) / SIGMA_THEO * 100.0

    print("\n" + "="*50)
    print("ANÁLISE MACROSCÓPICA DA TENSÃO SUPERFICIAL")
    print("="*50)
    print(f"Sigma Teórico (Integral C-H) : {SIGMA_THEO:.6e}")
    print(f"Sigma Numérico (Regressão)   : {sigma_num:.6e}")
    print(f"Erro Relativo do Sigma       : {erro_sigma:.2f} %")
    print(f"Pressão Espúria Residual     : {p_spurious:.6e}")

    # Renderização
    plt.style.use('default')
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    plt.figure(figsize=(7, 5))

    x_plot = np.linspace(0, max(curvaturas)*1.1, 100)
    plt.plot(x_plot, SIGMA_THEO * x_plot, 'k--', label=fr'Teórico ($\sigma_{{theo}}={SIGMA_THEO:.4e}$)')
    plt.plot(x_plot, sigma_num * x_plot + p_spurious, 'r-', label=fr'Numérico LBM ($\sigma_{{num}}={sigma_num:.4e}$)')
    plt.plot(curvaturas, delta_ps, 'ko', markersize=7, markerfacecolor='none', markeredgewidth=1.5, label='Dados VTK')

    plt.title('Validação da Lei de Laplace-Young', fontsize=12)
    plt.xlabel(r'Curvatura da Interface, $\frac{1}{R}$ [lu$^{-1}$]')
    plt.ylabel(r'Salto de Pressão, $\Delta P$ [lu]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("laplace_regression_acumulada.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    process_all_results()