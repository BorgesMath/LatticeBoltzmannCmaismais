#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "multiphase/cahn_hilliard.cuh"
#include "Magnetismo/poisson.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"
#include "lbm/lbm.cuh"

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

void check_numerical_stability() {
    std::cout << "========================================" << std::endl;
    std::cout << "DIAGNOSTICO DE ESTABILIDADE LBM" << std::endl;
    std::cout << "========================================" << std::endl;

    double cs = 1.0 / sqrt(3.0);
    double mach = U_INLET / cs;
    std::cout << "Numero de Mach (Inlet): " << mach;
    if (mach > 0.1) std::cout << " [ALERTA: Risco de erro de compressibilidade no BGK]" << std::endl;
    else std::cout << " [OK: Regime Incompressivel garantido]" << std::endl;

    double nu_in = (TAU_IN - 0.5) / 3.0;
    double nu_out = (TAU_OUT - 0.5) / 3.0;
    std::cout << "Viscosidade Cinematica (Fase 1): " << nu_in << std::endl;
    std::cout << "Viscosidade Cinematica (Fase 2): " << nu_out << std::endl;

    if (TAU_IN <= 0.5 || TAU_OUT <= 0.5) {
        std::cerr << "ERRO FATAL: Tempo de relaxacao <= 0.5 induz viscosidade negativa." << std::endl;
        exit(EXIT_FAILURE);
    }

    double cfl_ch = (M_MOBILITY * DT_CH) / 1.0;
    std::cout << "Numero CFL (Cahn-Hilliard): " << cfl_ch;
    if (cfl_ch > 0.1) std::cout << " [ALERTA: Integracao Explicita de Fase pode divergir]" << std::endl;
    else std::cout << " [OK]" << std::endl;

    if (IS_PERIODIC) std::cout << "\n[MODO DE OPERACAO]: Validacao Monofasica Periodica Ativa." << std::endl;
    else std::cout << "\n[MODO DE OPERACAO]: Producao Multifasica Saffman-Taylor Ativa." << std::endl;

    std::cout << "========================================\n" << std::endl;
}

void print_progress_bar(int step, int total, double omega_num, double omega_theo) {
    float progress = (float)step / total;
    int barWidth = 50;

    std::cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% "
              << "| It: " << std::setw(4) << step
              << " | w_num: " << std::showpos << std::scientific << std::setprecision(3) << omega_num
              << " | w_theo: " << omega_theo << std::noshowpos << "     ";
    std::cout.flush();

    if (step == total) std::cout << std::endl;
}

double calculate_amplitude(const double* h_phi) {
    double x_peak = 0.0;
    double x_valley = (double)NX;

    for (int y = 0; y < NY; ++y) {
        for (int x = 1; x < NX; ++x) {
            int idx = y * NX + x;
            int idx_prev = y * NX + (x - 1);

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

int main() {
    LBM_Populations d_f_in, d_f_out;
    Macro_Fields d_fields;
    size_t mem_size = NUM_NODES * sizeof(double);

    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    double *h_phi = (double*)malloc(mem_size);
    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    if (!h_phi || !h_rho || !h_ux || !h_uy) {
        std::cerr << "Falha de segmentacao na alocacao de RAM do Host." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string output_dir = init_post_processing();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields);
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

    check_numerical_stability();

    int max_iter = IS_PERIODIC ? 5000 : 50000;
    double chi_max = 1.2;

    double prev_amplitude = INITIAL_AMPLITUDE;
    double omega_num = 0.0;
    double omega_num_mid = 0.0;
    double omega_num_sum = 0.0;
    int omega_samples = 0;

    double k_wave = (2.0 * PI * MODE_M) / (double)NY;
    double omega_base = (k_wave * U_INLET * (TAU_OUT - TAU_IN) / 3.0) - (SIGMA * pow(k_wave, 3));
    double termo_magnetico = 0.0;
    double omega_theo = omega_base + termo_magnetico;

    for (int t = 0; t <= max_iter; ++t) {

        solve_cahn_hilliard(d_fields, numBlocks, threadsPerBlock);

        update_susceptibility_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, chi_max);
        CUDA_CHECK(cudaDeviceSynchronize());

        solve_poisson_magnetic(d_fields, numBlocks, threadsPerBlock);

        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (!IS_PERIODIC) {
            apply_open_boundaries(d_f_out, d_fields, NY);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        swap_populations(&d_f_in, &d_f_out);

        if (t % SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_phi, h_rho, h_ux, h_uy);

            if (!IS_PERIODIC) {
                double current_amplitude = calculate_amplitude(h_phi);
                if (t > 0 && current_amplitude > 0 && prev_amplitude > 0) {
                    omega_num = log(current_amplitude / prev_amplitude) / SNAPSHOT_STEPS;
                    if (t > max_iter * 0.1) {
                        omega_num_sum += omega_num;
                        omega_samples++;
                    }
                }
                prev_amplitude = current_amplitude;
            }
        }

        if (t == max_iter / 2) {
            omega_num_mid = omega_num;
        }

        if (t % 10 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter, omega_num, omega_theo);
        }
    }

    if (!IS_PERIODIC) {
        double omega_num_avg = (omega_samples > 0) ? (omega_num_sum / omega_samples) : 0.0;
        write_simulation_summary(output_dir, omega_theo, omega_num_mid, omega_num_avg);
    }

    free(h_phi); free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out); free_macro_fields(&d_fields);

    return 0;
}