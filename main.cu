#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstring>

#include <nlohmann/json.hpp>

#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "Magnetismo/poisson.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"
#include "lbm/lbm.cuh"
#include "multiphase/allen_cahn.cuh"

using json = nlohmann::json;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Erro CUDA na linha " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void allocate_populations(LBM_Populations* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->f0), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f2), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f4), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f6), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f8), bytes));
}

void allocate_populations_phase(LBM_Populations_Phase* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->g0), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->g1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->g2), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->g3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->g4), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->g5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->g6), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->g7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->g8), bytes));
}

void allocate_macro_fields(Macro_Fields* f, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(f->phi), bytes)); CUDA_CHECK(cudaMalloc((void**)&(f->phi_new), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->mu), bytes));  CUDA_CHECK(cudaMalloc((void**)&(f->psi), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->rho), bytes)); CUDA_CHECK(cudaMalloc((void**)&(f->ux), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->uy), bytes));  CUDA_CHECK(cudaMalloc((void**)&(f->chi_field), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->K_field), bytes));
}

void free_populations(LBM_Populations* p) {
    CUDA_CHECK(cudaFree(p->f0)); CUDA_CHECK(cudaFree(p->f1)); CUDA_CHECK(cudaFree(p->f2));
    CUDA_CHECK(cudaFree(p->f3)); CUDA_CHECK(cudaFree(p->f4)); CUDA_CHECK(cudaFree(p->f5));
    CUDA_CHECK(cudaFree(p->f6)); CUDA_CHECK(cudaFree(p->f7)); CUDA_CHECK(cudaFree(p->f8));
}

void free_populations_phase(LBM_Populations_Phase* p) {
    CUDA_CHECK(cudaFree(p->g0)); CUDA_CHECK(cudaFree(p->g1)); CUDA_CHECK(cudaFree(p->g2));
    CUDA_CHECK(cudaFree(p->g3)); CUDA_CHECK(cudaFree(p->g4)); CUDA_CHECK(cudaFree(p->g5));
    CUDA_CHECK(cudaFree(p->g6)); CUDA_CHECK(cudaFree(p->g7)); CUDA_CHECK(cudaFree(p->g8));
}

void free_macro_fields(Macro_Fields* f) {
    CUDA_CHECK(cudaFree(f->phi)); CUDA_CHECK(cudaFree(f->phi_new)); CUDA_CHECK(cudaFree(f->mu));
    CUDA_CHECK(cudaFree(f->psi)); CUDA_CHECK(cudaFree(f->rho)); CUDA_CHECK(cudaFree(f->ux));
    CUDA_CHECK(cudaFree(f->uy));  CUDA_CHECK(cudaFree(f->chi_field)); CUDA_CHECK(cudaFree(f->K_field));
}

void swap_populations(LBM_Populations* p1, LBM_Populations* p2) {
    LBM_Populations temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

void swap_populations_phase(LBM_Populations_Phase* p1, LBM_Populations_Phase* p2) {
    LBM_Populations_Phase temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

double calculate_amplitude(const double* h_phi, const SimConfig& cfg) {
    double x_peak = 0.0;
    double x_valley = (double)cfg.NX;

    for (int y = 0; y < cfg.NY; ++y) {
        for (int x = 1; x < cfg.NX; ++x) {
            int idx = y * cfg.NX + x;
            int idx_prev = y * cfg.NX + (x - 1);

            if (h_phi[idx_prev] > 0.0 && h_phi[idx] <= 0.0) {
                double w = h_phi[idx_prev] / (h_phi[idx_prev] - h_phi[idx]);
                double x_interface = (x - 1) + w;

                if (x_interface > x_peak) x_peak = x_interface;
                if (x_interface < x_valley) x_valley = x_interface;
                break;
            }
        }
    }
    return (x_peak - x_valley) / 2.0;
}

void print_progress_bar(int step, int total, double omega_num, double omega_theo) {
    float progress = (float)step / total;
    int barWidth = 50;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < int(barWidth * progress)) std::cout << "=";
        else if (i == int(barWidth * progress)) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% "
              << "| It: " << step
              << " | w_num: " << std::scientific << std::setprecision(3) << omega_num
              << " | w_theo: " << omega_theo;
    std::cout.flush();
    if (step == total) std::cout << std::endl;
}

void run_simulation_case(const SimConfig& cfg) {
    std::cout << "\n>>> INICIANDO CASO: " << cfg.case_name << " <<<" << std::endl;

    LBM_Populations d_f_in, d_f_out;
    LBM_Populations_Phase d_g_in, d_g_out;
    Macro_Fields d_fields;
    size_t mem_size = cfg.NUM_NODES * sizeof(double);

    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_populations_phase(&d_g_in, mem_size);
    allocate_populations_phase(&d_g_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    double *h_phi = (double*)malloc(mem_size);
    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    if (!h_phi || !h_rho || !h_ux || !h_uy) {
        std::cerr << "Falha de segmentacao na alocacao de RAM." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string output_dir = init_post_processing(cfg);

    std::ofstream csv_file(output_dir + "/growth_history.csv");
    csv_file << "TimeStep,Amplitude,Omega_Num,Mass,PhiMin,PhiMax\n";

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cfg.NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (cfg.NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_g_in, d_fields, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_f_out.f0, d_f_in.f0, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f1, d_f_in.f1, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f2, d_f_in.f2, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f3, d_f_in.f3, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f4, d_f_in.f4, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f5, d_f_in.f5, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f6, d_f_in.f6, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f7, d_f_in.f7, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f8, d_f_in.f8, mem_size, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(d_g_out.g0, d_g_in.g0, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g1, d_g_in.g1, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g2, d_g_in.g2, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g3, d_g_in.g3, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g4, d_g_in.g4, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g5, d_g_in.g5, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g6, d_g_in.g6, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g7, d_g_in.g7, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_out.g8, d_g_in.g8, mem_size, cudaMemcpyDeviceToDevice));

    int max_iter = 10000;
    double chi_max = 1.2;

    double prev_amplitude = cfg.INITIAL_AMPLITUDE;
    double omega_num = 0.0;

    std::vector<double> linear_omegas;

    double k_wave = (2.0 * PI * cfg.MODE_M) / (double)cfg.NY;
    double nu_in = (cfg.TAU_IN - 0.5) / 3.0;
    double nu_out = (cfg.TAU_OUT - 0.5) / 3.0;

    double den = (nu_in / cfg.K_0) + (nu_out / cfg.K_0);
    double viscous_forcing = (cfg.U_INLET / cfg.K_0) * (nu_out - nu_in);
    double capillary_forcing = cfg.SIGMA * k_wave * k_wave;
    double omega_base = (k_wave / den) * (viscous_forcing - capillary_forcing);

    double omega_theo = omega_base;

    double lambda = (double)cfg.NY / cfg.MODE_M;
    double max_linear_amplitude = 0.1 * lambda;
    bool in_linear_regime = true;

    for (int t = 0; t <= max_iter; ++t) {
        compute_chemical_potential_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, cfg);
        lbm_collide_and_stream_phase<<<numBlocks, threadsPerBlock>>>(d_g_in, d_g_out, d_fields, cfg);
        update_macroscopic_phase_kernel<<<numBlocks, threadsPerBlock>>>(d_g_out, d_fields, cfg);

        update_susceptibility_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, chi_max, cfg);
        solve_poisson_magnetic(d_fields, numBlocks, threadsPerBlock, cfg);

        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields, cfg);

        apply_open_boundaries(d_f_out, d_g_out, d_fields, cfg.NY, cfg);

        swap_populations(&d_f_in, &d_f_out);
        swap_populations_phase(&d_g_in, &d_g_out);

        if (t % cfg.SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_phi, h_rho, h_ux, h_uy, cfg);
            double current_amplitude = calculate_amplitude(h_phi, cfg);

            // CPU-side reduction da massa para evitar latência CUDA
            double total_mass = 0.0;
            double phi_min = h_phi[0];
            double phi_max = h_phi[0];

            for (int i = 0; i < cfg.NUM_NODES; ++i) {
                double val = h_phi[i];
                total_mass += val;
                if (val < phi_min) phi_min = val;
                if (val > phi_max) phi_max = val;
            }

            if (t > 0 && current_amplitude > 0 && prev_amplitude > 0) {
                omega_num = log(current_amplitude / prev_amplitude) / cfg.SNAPSHOT_STEPS;
                csv_file << t << "," << current_amplitude << "," << std::scientific << omega_num
                         << "," << total_mass << "," << phi_min << "," << phi_max << "\n";

                if (t > 1500 && in_linear_regime) {
                    if (current_amplitude < max_linear_amplitude) {
                        linear_omegas.push_back(omega_num);
                    } else {
                        in_linear_regime = false;
                    }
                }
            } else {
                csv_file << t << "," << current_amplitude << ",0.0"
                         << "," << total_mass << "," << phi_min << "," << phi_max << "\n";
            }
            prev_amplitude = current_amplitude;
        }

        if (t % 50 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter, omega_num, omega_theo);
        }
    }

    csv_file.close();

    double omega_plateau = 0.0;
    if (!linear_omegas.empty()) {
        int tail_count = std::min((int)linear_omegas.size(), 5);
        double sum = 0.0;
        for (int i = (int)linear_omegas.size() - tail_count; i < (int)linear_omegas.size(); ++i) {
            sum += linear_omegas[i];
        }
        omega_plateau = sum / tail_count;
    } else {
        omega_plateau = omega_num;
    }

    write_simulation_summary(output_dir, omega_theo, 0.0, omega_plateau);

    free(h_phi); free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out);
    free_populations_phase(&d_g_in); free_populations_phase(&d_g_out);
    free_macro_fields(&d_fields);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <arquivo_casos.json>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "ERRO FATAL: Nao foi possivel abrir o arquivo JSON." << std::endl;
        return EXIT_FAILURE;
    }

    json j_casos;
    file >> j_casos;
    std::vector<SimConfig> batch_configs;

    for (const auto& item : j_casos) {
        SimConfig cfg;
        std::string temp_name = item.value("case_name", "default");
        strncpy(cfg.case_name, temp_name.c_str(), sizeof(cfg.case_name) - 1);
        cfg.case_name[sizeof(cfg.case_name) - 1] = '\0';

        cfg.NX = item.at("NX");
        cfg.NY = item.at("NY");
        cfg.NUM_NODES = cfg.NX * cfg.NY;
        cfg.SNAPSHOT_STEPS = item.at("SNAPSHOT_STEPS");
        cfg.TAU_IN = item.at("TAU_IN");
        cfg.TAU_OUT = item.at("TAU_OUT");
        cfg.U_INLET = item.at("U_INLET");
        cfg.K_0 = item.at("K_0");
        cfg.M_MOBILITY = item.at("M_MOBILITY");
        cfg.CH_SUBSTEPS = item.at("CH_SUBSTEPS");
        cfg.DT_CH = 1.0 / (double)cfg.CH_SUBSTEPS;
        cfg.SIGMA = item.at("SIGMA");
        cfg.INTERFACE_WIDTH = item.at("INTERFACE_WIDTH");
        cfg.BETA = (3.0 * cfg.SIGMA) / (4.0 * cfg.INTERFACE_WIDTH);
        cfg.KAPPA = 3.0 * cfg.SIGMA * cfg.INTERFACE_WIDTH / 8.0;
        cfg.H0 = item.at("H0");
        cfg.H_ANGLE = item.at("H_ANGLE");
        cfg.SOR_OMEGA = item.at("SOR_OMEGA");
        cfg.SOR_ITERATIONS = item.at("SOR_ITERATIONS");
        cfg.INITIAL_AMPLITUDE = item.at("INITIAL_AMPLITUDE");
        cfg.MODE_M = item.at("MODE_M");
        cfg.BODY_FORCE_X = item.at("BODY_FORCE_X");
        cfg.IS_PERIODIC = item.at("IS_PERIODIC");

        batch_configs.push_back(cfg);
    }

    for (const auto& cfg : batch_configs) {
        run_simulation_case(cfg);
    }

    std::cout << "\n>>> EXECUCAO EM LOTE CONCLUIDA <<<" << std::endl;
    return 0;
}