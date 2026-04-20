// post_process/post_process.cu
#include "post_process.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

std::string init_post_processing() {
    // 1. Geração do Carimbo de Tempo (Timestamp)
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "results_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();

    // 2. Criação do Diretório
    if (!fs::create_directory(dir_name)) {
        std::cerr << "ERRO FATAL: Falha ao criar o diretorio de output " << dir_name << std::endl;
        exit(EXIT_FAILURE);
    }

    // 3. Serialização dos Parâmetros Estáticos (config_log.txt)
    std::ofstream config_file(dir_name + "/config_log.txt");
    if (config_file.is_open()) {
        config_file << "========================================\n";
        config_file << "LATTICE BOLTZMANN - LOG DE PARAMETROS\n";
        config_file << "========================================\n\n";

        config_file << "[TOPOLOGIA]\n";
        config_file << "NX = " << NX << "\nNY = " << NY << "\n\n";

        config_file << "[HIDRODINAMICA]\n";
        config_file << "TAU_IN = " << TAU_IN << "\nTAU_OUT = " << TAU_OUT << "\n";
        config_file << "K_0 = " << K_0 << "\nU_INLET = " << U_INLET << "\n\n";

        config_file << "[CAHN-HILLIARD]\n";
        config_file << "M_MOBILITY = " << M_MOBILITY << "\nSIGMA = " << SIGMA << "\n";
        config_file << "INTERFACE_WIDTH = " << INTERFACE_WIDTH << "\n";
        config_file << "CH_SUBSTEPS = " << CH_SUBSTEPS << "\n\n";

        config_file << "[MAGNETOSTATICA]\n";
        config_file << "H0 = " << H0 << "\nH_ANGLE = " << H_ANGLE << "\n";
        config_file << "SOR_OMEGA = " << SOR_OMEGA << "\n";

        config_file.close();
    }

    return dir_name;
}

void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields,
                double* h_phi, double* h_rho, double* h_ux, double* h_uy) {
    size_t mem_size = NUM_NODES * sizeof(double);

    cudaMemcpy(h_phi, d_fields.phi, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rho, d_fields.rho, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux, d_fields.ux, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy, d_fields.uy, mem_size, cudaMemcpyDeviceToHost);

    char filename[256];
    sprintf(filename, "%s/data_%06d.vtk", out_dir.c_str(), step);

    std::ofstream file(filename);
    if (!file.is_open()) return;

    // A MÁGICA: Gravar o ficheiro com precisão científica absoluta de 15 casas!
    file << std::scientific << std::setprecision(15);

    file << "# vtk DataFile Version 3.0\n";
    file << "LBM Data\n";
    file << "ASCII\n";
    file << "DATASET RECTILINEAR_GRID\n";
    file << "DIMENSIONS " << NX << " " << NY << " 1\n";

    file << "X_COORDINATES " << NX << " int\n";
    for (int i = 0; i < NX; ++i) file << i << " "; file << "\n";
    file << "Y_COORDINATES " << NY << " int\n";
    for (int j = 0; j < NY; ++j) file << j << " "; file << "\n";
    file << "Z_COORDINATES 1 int\n0\n";

    file << "POINT_DATA " << NUM_NODES << "\n";

    file << "SCALARS rho double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            file << h_rho[j * NX + i] << "\n";
        }
    }

    file << "SCALARS phi double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            file << h_phi[j * NX + i] << "\n";
        }
    }

    file << "VECTORS velocity double\n";
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            int idx = j * NX + i;
            file << h_ux[idx] << " " << h_uy[idx] << " 0.0\n";
        }
    }

    file.close();
}






// (Mantenha as funções init_post_processing e export_vtk intactas)

void write_simulation_summary(const std::string& out_dir, double omega_theo,
                              double omega_num_mid, double omega_num_avg) {
    std::ofstream config_file(out_dir + "/config_log.txt", std::ios_base::app);
    if (config_file.is_open()) {
        config_file << "\n========================================\n";
        config_file << "[ANALISE DE ESTABILIDADE (LSA E NUMERICA)]\n";
        config_file << "========================================\n\n";

        config_file << "-> PREDICAO TEORICA (Saffman-Taylor + Magnetismo):\n";
        config_file << "Taxa de Crescimento (w_theo) : " << std::scientific << omega_theo << "\n";

        if (omega_theo > 0.0) {
            config_file << "Regime Teorico Esperado      : INSTAVEL (Crescimento Viscoso)\n\n";
        } else {
            config_file << "Regime Teorico Esperado      : ESTAVEL (Supressao Capilar/Magnetica)\n\n";
        }

        config_file << "-> RESULTADOS NUMERICOS (Extracao LBM):\n";
        config_file << "Taxa no Meio da Simulacao (w_mid)  : " << std::scientific << omega_num_mid << "\n";
        config_file << "Taxa Media Assintotica (w_avg)     : " << std::scientific << omega_num_avg << "\n";

        // Avaliação de convergência
        double erro_relativo = std::abs((omega_num_avg - omega_theo) / omega_theo) * 100.0;
        config_file << "Desvio Numerico-Teorico Medio      : " << std::fixed << std::setprecision(2) << erro_relativo << " %\n";

        config_file << "========================================\n";
        config_file.close();
    }
}
    void save_metadata_laplace(const std::string& out_dir, double R0) {
        std::ofstream file(out_dir + "/config.txt");
        if (file.is_open()) {
            file << R0 << "\n";
            file.close();
        }
}