#include "post_process.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

std::string init_post_processing(const SimConfig& cfg) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << cfg.case_name << "_res_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();

    if (!fs::create_directory(dir_name)) {
        std::cerr << "ERRO FATAL: Falha ao criar o diretorio " << dir_name << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ofstream config_file(dir_name + "/config_log.txt");
    if (config_file.is_open()) {
        config_file << "========================================\n";
        config_file << "LATTICE BOLTZMANN - LOG DE PARAMETROS\n";
        config_file << "CASO: " << cfg.case_name << "\n";
        config_file << "========================================\n\n";

        config_file << "[TOPOLOGIA]\n";
        config_file << "NX = " << cfg.NX << "\nNY = " << cfg.NY << "\n\n";

        config_file << "[HIDRODINAMICA]\n";
        config_file << "TAU_IN = " << cfg.TAU_IN << "\nTAU_OUT = " << cfg.TAU_OUT << "\n";
        config_file << "K_0 = " << cfg.K_0 << "\nU_INLET = " << cfg.U_INLET << "\n\n";

        config_file << "[CAHN-HILLIARD]\n";
        config_file << "M_MOBILITY = " << cfg.M_MOBILITY << "\nSIGMA = " << cfg.SIGMA << "\n";
        config_file << "INTERFACE_WIDTH = " << cfg.INTERFACE_WIDTH << "\n";
        config_file << "CH_SUBSTEPS = " << cfg.CH_SUBSTEPS << "\n\n";

        config_file << "[MAGNETOSTATICA]\n";
        config_file << "H0 = " << cfg.H0 << "\nH_ANGLE = " << cfg.H_ANGLE << "\n";
        config_file << "SOR_OMEGA = " << cfg.SOR_OMEGA << "\n";

        config_file.close();
    }
    return dir_name;
}

void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields,
                double* h_phi, double* h_rho, double* h_ux, double* h_uy, const SimConfig& cfg) {

    size_t mem_size = cfg.NUM_NODES * sizeof(double);

    cudaMemcpy(h_phi, d_fields.phi, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rho, d_fields.rho, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux, d_fields.ux, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy, d_fields.uy, mem_size, cudaMemcpyDeviceToHost);

    std::stringstream ss;
    ss << out_dir << "/data_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    std::ofstream vtk(ss.str());

    if (vtk.is_open()) {
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "LBM_Data\nASCII\nDATASET STRUCTURED_POINTS\n";
        vtk << "DIMENSIONS " << cfg.NX << " " << cfg.NY << " 1\n";
        vtk << "ORIGIN 0 0 0\nSPACING 1 1 1\n";
        vtk << "POINT_DATA " << cfg.NUM_NODES << "\n";

        vtk << "SCALARS phi double 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < cfg.NUM_NODES; ++i) vtk << h_phi[i] << "\n";

        vtk << "SCALARS rho double 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < cfg.NUM_NODES; ++i) vtk << h_rho[i] << "\n";

        vtk << "VECTORS velocity double\n";
        for (int i = 0; i < cfg.NUM_NODES; ++i) vtk << h_ux[i] << " " << h_uy[i] << " 0.0\n";
        vtk.close();
    }
}

void write_simulation_summary(const std::string& out_dir, double omega_theo,
                              double omega_num_mid, double omega_num_avg) {
    std::ofstream config_file(out_dir + "/config_log.txt", std::ios_base::app);
    if (config_file.is_open()) {
        config_file << "\n========================================\n";
        config_file << "[ANALISE DE ESTABILIDADE]\n";
        config_file << "w_theo : " << std::scientific << omega_theo << "\n";
        config_file << "w_avg  : " << std::scientific << omega_num_avg << "\n";
        config_file << "========================================\n";
        config_file.close();
    }
}