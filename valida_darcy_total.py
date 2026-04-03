import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os

# Parâmetros fixos do config.cuh (Ajuste conforme sua simulação)
K_0_ANALITICO = 10.0
TAU = 1.0
NU = (TAU - 0.5) / 3.0

def get_latest_results():
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
    latest_dir = max(glob.glob(os.path.join(base_dir, "results_*")), key=os.path.getmtime)
    latest_vtk = max(glob.glob(os.path.join(latest_dir, "data_*.vtk")), key=os.path.getmtime)
    return latest_vtk

def run_advanced_validation():
    vtk_path = get_latest_results()
    mesh = pv.read(vtk_path)
    NX, NY, _ = mesh.dimensions

    # Extração de campos (vetorizados)
    rho = mesh.point_data['rho']
    ux = mesh.point_data['velocity'][:, 0]

    # --- TESTE 1: BALANÇO INTEGRAL DE MOMENTO ---
    # Como BODY_FORCE_X não está no VTK, usamos o valor do config.cuh
    # Se você mudou o config, altere aqui:
    G = 1e-5

    # Força Motriz Total (Integral de G * rho sobre o domínio)
    total_driving_force = np.sum(rho * G)

    # Força de Arrasto Total (Integral de rho * (nu/K) * u)
    total_drag_force = np.sum(rho * (NU / K_0_ANALITICO) * ux)

    resíduo = abs(total_driving_force - total_drag_force)
    erro_rel_conservacao = (resíduo / total_driving_force) * 100

    # --- TESTE 2: PERMEABILIDADE NUMÉRICA ---
    ux_avg = np.mean(ux)
    K_num = (ux_avg * NU) / G
    erro_K = abs(K_num - K_0_ANALITICO) / K_0_ANALITICO * 100

    print(f"\n--- RELATÓRIO DE CONSISTÊNCIA ---")
    print(f"Arquivo Analisado        : {os.path.basename(vtk_path)}")
    print(f"Balanço de Forças (Net)  : {resíduo:.6e}")
    print(f"Erro de Conservação      : {erro_rel_conservacao:.6f} %")
    print(f"Permeabilidade LBM (K)   : {K_num:.6f}")
    print(f"Erro vs K_analítico      : {erro_K:.4f} %")

    # --- VISUALIZAÇÃO DA LINEARIDADE (EXEMPLO DE DADOS) ---
    # Para usar isso, insira os resultados de diferentes simulações manuais:
    forces = [1e-6, 5e-6, 1e-5, 5e-5] # Exemplos de G
    velocities = [9.98e-7, 4.99e-6, 9.97e-6, 4.98e-5] # Exemplos de ux_avg medidos

    if len(forces) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(forces, velocities, 'ks', label='Dados LBM (GPU)')

        # Ajuste linear (deve passar pela origem)
        m, b = np.polyfit(forces, velocities, 1)
        plt.plot(forces, m*np.array(forces) + b, 'r--', label=f'Ajuste Linear (R²~1.0)')

        plt.title("Teste de Linearidade de Darcy (Invariância de Escala)")
        plt.xlabel("Força de Corpo (G) [lu/ts²]")
        plt.ylabel("Velocidade Média ($\langle u_x \\rangle$) [lu/ts]")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.show()

if __name__ == '__main__':
    run_advanced_validation()