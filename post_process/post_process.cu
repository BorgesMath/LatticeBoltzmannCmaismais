#include "post_process.cuh"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

std::string init_post_processing() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "results_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();

    if (!fs::create_directory(dir_name)) {
        std::cerr << "ERRO: Falha ao criar o diretorio " << dir_name << std::endl;
        exit(EXIT_FAILURE);
    }
    return dir_name;
}

// Assinatura limpa, exigindo apenas rho e velocidades
void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields, double* h_rho, double* h_ux, double* h_uy) {
    size_t mem_size = NUM_NODES * sizeof(double);
    cudaMemcpy(h_rho, d_fields.rho, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux, d_fields.ux, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy, d_fields.uy, mem_size, cudaMemcpyDeviceToHost);

    std::stringstream ss;
    ss << out_dir << "/data_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    std::ofstream vtk(ss.str());

    if (vtk.is_open()) {
        vtk << "# vtk DataFile Version 3.0\nLBM_Poiseuille_Data\nASCII\nDATASET STRUCTURED_POINTS\n";
        vtk << "DIMENSIONS " << NX << " " << NY << " 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA " << NUM_NODES << "\n";

        vtk << "SCALARS rho double 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < NUM_NODES; ++i) vtk << h_rho[i] << "\n";

        vtk << "VECTORS velocity double\n";
        for (int i = 0; i < NUM_NODES; ++i) vtk << h_ux[i] << " " << h_uy[i] << " 0.0\n";
        vtk.close();
    }
}