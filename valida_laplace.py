import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import laplace

def extract_parameters_from_log(log_path):
    """Lê parâmetros diretamente do config_log.txt gerado pelo C++"""
    params = {}
    if not os.path.exists(log_path):
        return params
    with open(log_path, 'r') as file:
        for line in file:
            if "=" in line:
                parts = line.split("=")
                key = parts[0].strip()
                val_str = parts[1].strip()
                try:
                    params[key] = float(val_str)
                except ValueError:
                    pass
    return params

def validate_laplace():
    base_dir = r"C:\Users\mathe\CLionProjects\LatticeBoltzmann\cmake-build-release"

    # 1. LOCALIZAÇÃO DINÂMICA DA ÚLTIMA SIMULAÇÃO
    results_dirs = glob.glob(os.path.join(base_dir, "results_*"))
    if not results_dirs:
        print("Nenhum diretório 'results_*' encontrado.")
        return
    latest_dir = max(results_dirs, key=os.path.getmtime)

    # Extração de parâmetros para evitar inserção manual
    log_path = os.path.join(latest_dir, "config_log.txt")
    params = extract_parameters_from_log(log_path)

    if "SIGMA" not in params:
        print(f"Erro: Parâmetros (SIGMA/INTERFACE_WIDTH) não encontrados em {log_path}")
        return

    sigma_theo = params["SIGMA"]
    w_int = params["INTERFACE_WIDTH"]

    # Mapeamento rigoroso de Ginzburg-Landau para garantir W_num = INTERFACE_WIDTH
    beta = (3.0 * sigma_theo) / (8.0 * w_int)
    kappa = (3.0 * sigma_theo * w_int) / 4.0

    # 2. LEITURA DO ÚLTIMO VTK
    vtk_files = glob.glob(os.path.join(latest_dir, "data_*.vtk"))
    if not vtk_files:
        print(f"Nenhum arquivo VTK encontrado em {latest_dir}")
        return
    latest_vtk = max(vtk_files, key=os.path.getmtime)

    mesh = pv.read(latest_vtk)
    NX, NY, _ = mesh.dimensions
    rho_field = mesh.point_data['rho'].reshape((NY, NX))
    phi_field = mesh.point_data['phi'].reshape((NY, NX))

    # 3. CÁLCULO DA PRESSÃO TERMODINÂMICA (EQUAÇÃO DE ESTADO NÃO-IDEAL)
    # P_th = rho*cs^2 + f(phi) - kappa*phi*laplace(phi)
    lap_phi = laplace(phi_field, mode='wrap') # Periódico
    p_mech = rho_field / 3.0

    # Contribuição do potencial de poço-duplo
    free_energy_bulk = beta * (phi_field**2 - 1.0) * (3.0 * phi_field**2 + 1.0)
    p_th_field = p_mech + free_energy_bulk - kappa * phi_field * lap_phi

    # 4. CÁLCULO DE RAIO E PRESSÕES EM REGIÕES PURAS
    # Raio Efetivo via integral de área
    area_fase = np.sum(phi_field > 0.0)
    raio_num = np.sqrt(area_fase / np.pi)

    # Máscaras de Regiões Puras para ignorar a zona difusa da interface [cite: 55, 56]
    mask_in = phi_field > 0.95
    mask_out = phi_field < -0.95

    p_in = np.mean(p_th_field[mask_in])
    p_out = np.mean(p_th_field[mask_out])
    delta_p_num = p_in - p_out

    # 5. COMPARAÇÃO COM A LEI DE LAPLACE-YOUNG
    delta_p_teorico = sigma_theo / raio_num
    erro_relativo = abs(delta_p_num - delta_p_teorico) / delta_p_teorico * 100.0

    print("-" * 60)
    print(f"VALIDAÇÃO DA LEI DE LAPLACE (LBM MULTIFÁSICO)")
    print(f"Diretório: {os.path.basename(latest_dir)}")
    print("-" * 60)
    print(f"Parâmetros: SIGMA={sigma_theo:.2e} | W={w_int:.2f} | BETA={beta:.4e} | KAPPA={kappa:.4e}")
    print(f"Raio Numérico (R)       : {raio_num:.4f} lu")
    print(f"P_inside (Média Pura)   : {p_in:.6e} lu")
    print(f"P_outside (Média Pura)  : {p_out:.6e} lu")
    print(f"Delta P Numérico (DP)   : {delta_p_num:.6e} lu")
    print(f"Delta P Teórico (s/R)   : {delta_p_teorico:.6e} lu")
    print(f"Erro Relativo           : {erro_relativo:.4f} %")
    print("-" * 60)

if __name__ == '__main__':
    validate_laplace()