import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

# Configurações globais de plotagem (Padrão Acadêmico/Artigo)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True
})

# Condição de Contorno Cinemática (Dirichlet) do config.cuh
U_INLET = 0.01

def get_latest_vtk_file():
    """Identifica e retorna o arquivo de dados (VTK) mais recente gerado pelo solver."""
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
    result_dirs = glob.glob(os.path.join(base_dir, "results_*"))

    if not result_dirs:
        raise FileNotFoundError(f"Diretório de resultados não encontrado em {base_dir}")

    latest_dir = max(result_dirs, key=os.path.getmtime)
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))

    if not vtk_files:
        raise FileNotFoundError(f"Arquivo de malha VTK ausente no diretório {latest_dir}")

    return max(vtk_files, key=os.path.getmtime)

def validate_poiseuille_incompressible():
    """Validação LBM para formulação incompressível restrita à topologia Half-Way."""
    vtk_file = get_latest_vtk_file()
    print(f"Processando tensor do arquivo: {os.path.basename(vtk_file)}")

    # Leitura Topológica da malha Euleriana
    mesh = pv.read(vtk_file)
    NX, NY, _ = mesh.dimensions

    # Extração de Campos e Remapeamento Matricial (2D)
    ux_field = mesh.point_data['velocity'][:, 0]
    ux_2d = ux_field.reshape((NY, NX))

    # Extração Transversal (Regime Desenvolvido)
    x_eval = NX - 10
    ux_num = ux_2d[:, x_eval]
    y_coords = np.arange(NY)

    # 1. Correção Topológica da Condição de Contorno (Half-Way Bounce-Back)
    H_real = NY
    y_fisico = y_coords + 0.5

    # 2. Formulação Analítica Teórica (Modelo Incompressível)
    U_max_ideal = 1.5 * U_INLET
    ux_ana = (4.0 * U_max_ideal / (H_real**2)) * y_fisico * (H_real - y_fisico)

    # Vetores para plotagem contínua do referencial teórico
    y_plot = np.linspace(0, H_real, 200)
    ux_plot = (4.0 * U_max_ideal / (H_real**2)) * y_plot * (H_real - y_plot)

    # 3. Métricas de Convergência (L2, MAE, RMSE)
    erro_l2 = np.linalg.norm(ux_num - ux_ana) / np.linalg.norm(ux_ana)
    mae = np.mean(np.abs(ux_num - ux_ana))
    rmse = np.sqrt(np.mean((ux_num - ux_ana)**2))

    print(f"\n==================================================")
    print(f"ANÁLISE DE CONVERGÊNCIA: MODELO INCOMPRESSÍVEL")
    print(f"Seção Avaliada           : x = {x_eval}")
    print(f"==================================================")
    print(f"Velocidade Inlet Implic. : {U_INLET:.6f}")
    print(f"Velocidade Média Medida  : {np.mean(ux_num):.6f}")
    print(f"--------------------------------------------------")
    print(f"Erro Relativo (Norma L2) : {erro_l2:.4e}")
    print(f"Erro Absoluto (MAE)      : {mae:.4e}")
    print(f"Raiz Erro Quadr. (RMSE)  : {rmse:.4e}")
    print(f"==================================================\n")

    # 4. Renderização Gráfica Acadêmica
    fig, ax = plt.subplots(figsize=(6, 5)) # Proporção otimizada para colunas de artigos (IEEE/Elsevier)

    ax.plot(ux_plot, y_plot, 'k-', linewidth=1.5,
            label='Analytical')

    ax.plot(ux_num, y_fisico, 'ko', markeredgecolor='black', markerfacecolor='none',
            label='Numerical (LBM)', markersize=6, markeredgewidth=1.2)

    # Em artigos, o título é omitido em favor do "Figure Caption" no documento LaTeX/Word.
    ax.set_xlabel(r'Longitudinal velocity, $u_x$ [lu]')
    ax.set_ylabel(r'Transverse coordinate, $y$ [lu]')

    ax.set_ylim(0, H_real)
    ax.set_xlim(left=0.0) # Garante que o eixo X inicie em 0

    # Legenda sem bordas sólidas pesadas
    ax.legend(loc='best', frameon=False)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(vtk_file), "validacao_poiseuille_academic.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de validação gerado em: {plot_path}")

    plt.show()

if __name__ == '__main__':
    validate_poiseuille_incompressible()