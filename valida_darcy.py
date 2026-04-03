import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

# Parâmetros físicos do config.cuh
K_0 = 10.0
TAU = 1.0
BODY_FORCE_X = 1e-5

def validate_brinkman_darcy():
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
    result_dirs = glob.glob(os.path.join(base_dir, "results_*"))

    if not result_dirs:
        raise FileNotFoundError("Nenhum diretorio de resultados encontrado.")

    latest_dir = max(result_dirs, key=os.path.getmtime)
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))

    if not vtk_files:
        raise FileNotFoundError(f"Nenhum arquivo VTK encontrado em {latest_dir}")

    latest_vtk = max(vtk_files, key=os.path.getmtime)

    mesh = pv.read(latest_vtk)
    NX, NY, _ = mesh.dimensions

    print(f"[{os.path.basename(latest_vtk)}] Topologia extraida: NX = {NX}, NY = {NY}")

    # Extração de Campos 2D
    ux_field = mesh.point_data['velocity'][:, 0].reshape((NY, NX))
    rho_field = mesh.point_data['rho'].reshape((NY, NX))

    # Eixos de avaliação
    x_mid = NX // 2
    y_coords = np.arange(NY)
    x_coords = np.arange(NX)

    # 1. PERFIL DE VELOCIDADE (Eixo Y)
    ux_num = ux_field[:, x_mid]

    # Soluções Analíticas
    H = NY - 1
    nu = (TAU - 0.5) / 3.0

    # A) Brinkman-Darcy (Exato para o seu código atual com K_0)
    sqrt_K = np.sqrt(K_0)
    ux_brinkman = (K_0 / nu) * BODY_FORCE_X * (1.0 - np.cosh((y_coords - H/2) / sqrt_K) / np.cosh(H / (2 * sqrt_K)))

    # B) Poiseuille/Navier-Stokes (Limite de permeabilidade infinita)
    ux_poiseuille = (BODY_FORCE_X / (2 * nu)) * y_coords * (H - y_coords)

    # 2. PERFIL DE PRESSÃO (Eixo X)
    # LBM Equação de Estado: P = c_s^2 * rho
    p_num = (rho_field[NY//2, :] / 3.0)

    # Pressão Analítica Equivalente (Integrada a partir da Força de Corpo)
    p_ref = p_num[0]
    p_equiv = p_ref - BODY_FORCE_X * x_coords

    # Erro relativo no núcleo de Brinkman
    erro_rel = np.linalg.norm(ux_num - ux_brinkman) / np.linalg.norm(ux_brinkman) * 100.0
    print(f"Erro L2 (LBM vs Brinkman-Darcy): {erro_rel:.4e} %")

    # =======================================================
    # RENDERIZAÇÃO GRÁFICA MULTIPLOT
    # =======================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Cinemática (Velocidades)

    ax1.plot(ux_num, y_coords, 'ko', markeredgecolor='black', markerfacecolor='none', label='LBM Numérico', markersize=6)
    ax1.plot(ux_brinkman, y_coords, 'b-', linewidth=2, label=rf'Brinkman-Darcy Analítico ($K_0={K_0}$)')
    ax1.plot(ux_poiseuille, y_coords, 'r--', linewidth=2, label=r'Poiseuille Analítico ($K_0 \rightarrow \infty$)')
    ax1.set_title(f'Perfil Transversal de Velocidade (X = {x_mid})', fontsize=12)
    ax1.set_xlabel(r'Velocidade Longitudinal $u_x$ [lu]', fontsize=11)
    ax1.set_ylabel(r'Coordenada Transversal $y$ [lu]', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Subplot 2: Termodinâmica (Pressão)
    ax2.plot(x_coords, p_num, 'k-', linewidth=2, label='Pressão Numérica (LBM via $\\rho/3$)')
    ax2.plot(x_coords, p_equiv, 'r--', linewidth=2, label='Queda de Pressão Equivalente a $F_x$')
    ax2.set_title('Mecânica da Incompressibilidade Periódica (Eixo Central Y)', fontsize=12)
    ax2.set_xlabel(r'Coordenada Longitudinal $x$ [lu]', fontsize=11)
    ax2.set_ylabel(r'Pressão $P$ [lu]', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Ajuste de escala para destacar a isobárica numérica
    ax2.set_ylim(p_num[0] - BODY_FORCE_X * NX * 1.5, p_num[0] + BODY_FORCE_X * NX * 0.5)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(latest_vtk), "validacao_brinkman_darcy_pressao.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico de validação avançada salvo em: {plot_path}")

    plt.show()

if __name__ == '__main__':
    validate_brinkman_darcy()