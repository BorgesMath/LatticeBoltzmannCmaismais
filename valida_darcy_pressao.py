import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

# Padrão Acadêmico de Renderização
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

# Parâmetros Físicos definidos no config.cuh
K_IMPOSED = 10.0
TAU = 1.0
NU = (TAU - 0.5) / 3.0

def get_latest_vtk_file():
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
    result_dirs = glob.glob(os.path.join(base_dir, "results_*"))
    latest_dir = max(result_dirs, key=os.path.getmtime)
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))
    return max(vtk_files, key=os.path.getmtime)

def validate_darcy_permeability():
    vtk_file = get_latest_vtk_file()
    print(f"Reading dataset: {os.path.basename(vtk_file)}")

    mesh = pv.read(vtk_file)
    NX, NY, _ = mesh.dimensions

    # Remapeamento 2D dos campos
    ux_2d = mesh.point_data['velocity'][:, 0].reshape((NY, NX))
    rho_2d = mesh.point_data['rho'].reshape((NY, NX))
    p_2d = rho_2d / 3.0  # Equação de Estado LBM: P = rho * c_s^2

    # Extração no Eixo Central (Isolamento da camada de Brinkman)
    y_center = NY // 2
    x_coords = np.arange(NX)
    p_centerline = p_2d[y_center, :]
    u_centerline = ux_2d[y_center, :]

    # Seleção da Região de Bulk (Ignora 25% iniciais e finais para evitar efeitos de borda)
    bulk_start = int(0.25 * NX)
    bulk_end = int(0.75 * NX)
    x_bulk = x_coords[bulk_start:bulk_end]
    p_bulk = p_centerline[bulk_start:bulk_end]
    u_bulk = u_centerline[bulk_start:bulk_end]

    # 1. Recuperação do Gradiente de Pressão (Regressão Linear Mínimos Quadrados)
    coeffs = np.polyfit(x_bulk, p_bulk, 1)
    dp_dx = coeffs[0]
    p_fit = np.polyval(coeffs, x_coords)

    # 2. Recuperação da Permeabilidade K via Lei de Darcy
    u_avg_bulk = np.mean(u_bulk)
    rho_avg_bulk = np.mean(rho_2d[y_center, bulk_start:bulk_end])

    # K = (nu * rho * u) / (-dP/dx)
    K_recovered = (NU * rho_avg_bulk * u_avg_bulk) / (-dp_dx)
    erro_relativo = abs(K_recovered - K_IMPOSED) / K_IMPOSED * 100.0

    # 3. Solução Analítica do Perfil de Brinkman (Transversal) para o painel B
    y_coords = np.arange(NY)
    y_fisico = y_coords + 0.5
    H_real = NY

    # U_darcy = vazão teórica se não houvesse parede
    U_darcy = (-dp_dx) * K_IMPOSED / (NU * rho_avg_bulk)
    ux_ana = U_darcy * (1.0 - np.cosh((y_fisico - H_real/2) / np.sqrt(K_IMPOSED)) / np.cosh(H_real / (2 * np.sqrt(K_IMPOSED))))

    ux_num_transversal = ux_2d[:, bulk_end] # Perfil no final da zona de bulk

    # Relatório de Console
    print(f"\n==================================================")
    print(f"DARCY PERMEABILITY RECOVERY ANALYSIS")
    print(f"==================================================")
    print(f"Bulk Region Analyzed : x = {bulk_start} to {bulk_end}")
    print(f"Pressure Gradient    : {dp_dx:.6e} [lu/ts^2]")
    print(f"Centerline Velocity  : {u_avg_bulk:.6e} [lu/ts]")
    print(f"--------------------------------------------------")
    print(f"Imposed Permeability (K_0)   : {K_IMPOSED:.4f}")
    print(f"Recovered Permeability (K_r) : {K_recovered:.4f}")
    print(f"Relative Error               : {erro_relativo:.4f} %")
    print(f"==================================================\n")

    # Renderização Acadêmica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Painel A: Queda de Pressão
    ax1.plot(x_coords, p_centerline, 'k-', linewidth=1.5, label='LBM Pressure')
    ax1.plot(x_bulk, p_fit[bulk_start:bulk_end], 'r--', linewidth=2, label=f'Linear Fit ($\\nabla P$)')
    ax1.axvline(x=bulk_start, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=bulk_end, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel(r'Longitudinal coordinate, $x$ [lu]')
    ax1.set_ylabel(r'Thermodynamic Pressure, $P$ [lu]')
    ax1.legend(loc='best', frameon=False)
    ax1.text(0.05, 0.05, f'$K_{{recovered}} = {K_recovered:.2f}$', transform=ax1.transAxes,
             fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Painel B: Perfil de Brinkman (Plug Flow)
    ax2.plot(ux_ana, y_fisico, 'k-', linewidth=1.5, label='Brinkman Analytical')
    ax2.plot(ux_num_transversal, y_fisico, 'ko', markeredgecolor='black', markerfacecolor='none',
             label='Numerical (LBM)', markersize=5)
    ax2.set_xlabel(r'Longitudinal velocity, $u_x$ [lu]')
    ax2.set_ylabel(r'Transverse coordinate, $y$ [lu]')
    ax2.set_ylim(0, H_real)
    ax2.legend(loc='best', frameon=False)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(vtk_file), "darcy_validation_academic.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.show()

if __name__ == '__main__':
    validate_darcy_permeability()