import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

# Parâmetros espelhados do config.cuh
NX = 400
NY = 50
U_INLET = 0.01

def get_latest_vtk_file():
    # Caminho base do seu diretório de build
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"

    # Encontra todas as pastas 'results_' e pega a mais recente
    result_dirs = glob.glob(os.path.join(base_dir, "results_*"))
    if not result_dirs:
        raise FileNotFoundError(f"Nenhuma pasta de resultados encontrada em {base_dir}")

    latest_dir = max(result_dirs, key=os.path.getmtime)

    # Encontra todos os arquivos .vtk na pasta mais recente e pega o último (maior tempo)
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"Nenhum arquivo .vtk encontrado na pasta {latest_dir}")

    latest_vtk = max(vtk_files, key=os.path.getmtime)
    return latest_vtk

def validate_poiseuille():
    vtk_file = get_latest_vtk_file()
    print(f"Lendo dados numéricos do arquivo: {vtk_file}")

    # Leitura da malha Euleriana
    mesh = pv.read(vtk_file)
    ux_field = mesh.point_data['velocity'][:, 0]

    # Remapeamento topológico 1D -> 2D (NY, NX)
    # No VTK STRUCTURED_POINTS, o eixo X varia mais rápido, então a ordem de reshape é (NY, NX)
    ux_2d = ux_field.reshape((NY, NX))

    # Extração do perfil de velocidade transversal próximo à saída (Escoamento Desenvolvido)
    x_eval = NX - 10
    ux_num = ux_2d[:, x_eval]
    y_coords = np.arange(NY)

    # Formulação do Perfil Analítico (Poiseuille)
    # A condição de Bounce-Back coloca a parede matemática exatamente no nó.
    H = NY - 1
    U_max = 1.5 * U_INLET
    ux_ana = (4.0 * U_max / (H**2)) * y_coords * (H - y_coords)

    # Computação do Erro Relativo em Norma L2
    erro_l2 = np.linalg.norm(ux_num - ux_ana) / np.linalg.norm(ux_ana)
    print(f"Norma L2 do Erro Relativo (LBM vs Navier-Stokes): {erro_l2:.4e}")

    # Renderização da Validação
    plt.figure(figsize=(8, 6))
    plt.plot(ux_num, y_coords, 'o', markeredgecolor='black', markerfacecolor='none', label='LBM (Numérico GPU)')
    plt.plot(ux_ana, y_coords, 'r-', linewidth=2, label='Poiseuille (Teórico Analítico)')
    plt.title(f'Validação Hidrodinâmica LBM - Perfil na Seção X = {x_eval}')
    plt.xlabel(r'Velocidade $u_x$ [lu]')
    plt.ylabel(r'Coordenada Transversal $y$ [lu]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Salva o plot no mesmo diretório do arquivo VTK
    plot_path = os.path.join(os.path.dirname(vtk_file), "validacao_poiseuille.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico de validação salvo em: {plot_path}")

    plt.show()

if __name__ == '__main__':
    validate_poiseuille()