import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

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
    # U_mean ao longo de x é estritamente igual a U_INLET
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

    # 4. Renderização Comparativa
    plt.figure(figsize=(8, 6))

    plt.plot(ux_plot, y_plot, 'r-', linewidth=2,
             label='Solução Analítica Estacionária (Poiseuille)')

    plt.plot(ux_num, y_fisico, 'ko', markeredgecolor='black', markerfacecolor='none',
             label='Dinâmica LBM (Incompressível + Half-Way)', markersize=6)

    plt.title(f'Perfil Transversal de Momento Linear - Seção $x = {x_eval}$', fontsize=12)
    plt.xlabel(r'Velocidade Longitudinal $u_x$ [lu]', fontsize=11)
    plt.ylabel(r'Coordenada Física Transversal $y$ [lu]', fontsize=11)

    plt.ylim(0, H_real)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(vtk_file), "validacao_poiseuille_incompressivel.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico de validação gerado em: {plot_path}")

    plt.show()

if __name__ == '__main__':
    validate_poiseuille_incompressible()