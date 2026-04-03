import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os


# No início do valida_laplace.py
BETA = 0.01   # INSIRA O SEU VALOR EXATO DO config.cuh
KAPPA = 0.02  # INSIRA O SEU VALOR EXATO DO config.cuh

# Cálculo analítico rigoroso de Cahn-Hilliard
W_interface = np.sqrt(KAPPA / (2.0 * BETA))
SIGMA_TEORICA = (4.0 * np.sqrt(2.0) / 3.0) * np.sqrt(KAPPA * BETA)

print(f"Espessura Interfacial Numérica (W) : {W_interface:.4f} lu")
print(f"Tensão Superficial Intrínseca (s)  : {SIGMA_TEORICA:.6e}")



def validate_laplace():
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"
    latest_dir = max(glob.glob(os.path.join(base_dir, "results_*")), key=os.path.getmtime)
    latest_vtk = max(glob.glob(os.path.join(latest_dir, "data_*.vtk")), key=os.path.getmtime)

    mesh = pv.read(latest_vtk)
    NX, NY, _ = mesh.dimensions

    rho_field = mesh.point_data['rho'].reshape((NY, NX))
    phi_field = mesh.point_data['phi'].reshape((NY, NX))

    # 1. CÁLCULO DO RAIO NUMÉRICO EFETIVO
    # O raio real da gota no equilíbrio é a integral de área da fase isolada
    # A(phi > 0) = pi * R^2
    area_fase = np.sum(phi_field > 0.0)
    raio_num = np.sqrt(area_fase / np.pi)

    # 2. EXTRAÇÃO DE PRESSÕES (Afastado da zona difusa da interface)
    centro_y, centro_x = NY // 2, NX // 2

    # Pressão interna (média de um cluster no núcleo da gota)
    rho_in = np.mean(rho_field[centro_y-2:centro_y+3, centro_x-2:centro_x+3])
    p_in = rho_in / 3.0

    # Pressão externa (média nas extremidades do domínio periódico)
    rho_out = np.mean(rho_field[0:5, 0:5])
    p_out = rho_out / 3.0

    delta_p_num = p_in - p_out

    # 3. COMPARAÇÃO COM LEI DE LAPLACE
    delta_p_teorico = SIGMA_TEORICA / raio_num
    erro_relativo = abs(delta_p_num - delta_p_teorico) / delta_p_teorico * 100.0

    print("-" * 50)
    print("VALIDAÇÃO DA LEI DE LAPLACE (MULTIFÁSICO LBM)")
    print("-" * 50)
    print(f"Raio Efetivo de Equilíbrio (R) : {raio_num:.4f} lu")
    print(f"Salto de Pressão Numérico (DP) : {delta_p_num:.6e} lu")
    print(f"Salto de Pressão Teórico (s/R) : {delta_p_teorico:.6e} lu")
    print(f"Erro Relativo                  : {erro_relativo:.4f} %")
    print("-" * 50)

if __name__ == '__main__':
    validate_laplace()