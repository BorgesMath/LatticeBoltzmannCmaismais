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

    // Transferência Síncrona Device-to-Host (D2H)
    cudaMemcpy(h_phi, d_fields.phi, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rho, d_fields.rho, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux, d_fields.ux, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy, d_fields.uy, mem_size, cudaMemcpyDeviceToHost);

    // Formatação do nome do arquivo (ex: data_000120.vtk)
    std::stringstream ss;
    ss << out_dir << "/data_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    std::ofstream vtk(ss.str());

    if (vtk.is_open()) {
        // Cabeçalho VTK Legacy
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "LBM_Multiphase_Magnetic_Data\n";
        vtk << "ASCII\n";
        vtk << "DATASET STRUCTURED_POINTS\n";
        vtk << "DIMENSIONS " << NX << " " << NY << " 1\n";
        vtk << "ORIGIN 0 0 0\n";
        vtk << "SPACING 1 1 1\n";
        vtk << "POINT_DATA " << NUM_NODES << "\n";

        // Escalar: Parâmetro de Ordem (Fase)
        vtk << "SCALARS phi double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < NUM_NODES; ++i) vtk << h_phi[i] << "\n";

        // Escalar: Densidade Macroscópica
        vtk << "SCALARS rho double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < NUM_NODES; ++i) vtk << h_rho[i] << "\n";

        // Vetor: Campo de Velocidade Físico
        vtk << "VECTORS velocity double\n";
        for (int i = 0; i < NUM_NODES; ++i) vtk << h_ux[i] << " " << h_uy[i] << " 0.0\n";

        vtk.close();
    }
}