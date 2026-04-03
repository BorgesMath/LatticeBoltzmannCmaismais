#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Inclusões Modulares
#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "multiphase/cahn_hilliard.cuh"
#include "Magnetismo/poisson.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"

// Tratamento de falhas na API CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Erro CUDA na linha " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =========================================================
// GERENCIAMENTO DE MEMÓRIA DE VÍDEO (VRAM)
// =========================================================

void allocate_populations(LBM_Populations* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->f0), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f2), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f4), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f6), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f8), bytes));
}

void allocate_macro_fields(Macro_Fields* f, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(f->phi), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->phi_new), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->mu), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->psi), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->rho), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->ux), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->uy), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->chi_field), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->K_field), bytes));
}

void free_populations(LBM_Populations* p) {
    CUDA_CHECK(cudaFree(p->f0)); CUDA_CHECK(cudaFree(p->f1));
    CUDA_CHECK(cudaFree(p->f2)); CUDA_CHECK(cudaFree(p->f3));
    CUDA_CHECK(cudaFree(p->f4)); CUDA_CHECK(cudaFree(p->f5));
    CUDA_CHECK(cudaFree(p->f6)); CUDA_CHECK(cudaFree(p->f7));
    CUDA_CHECK(cudaFree(p->f8));
}

void free_macro_fields(Macro_Fields* f) {
    CUDA_CHECK(cudaFree(f->phi)); CUDA_CHECK(cudaFree(f->phi_new));
    CUDA_CHECK(cudaFree(f->mu));  CUDA_CHECK(cudaFree(f->psi));
    CUDA_CHECK(cudaFree(f->rho)); CUDA_CHECK(cudaFree(f->ux));
    CUDA_CHECK(cudaFree(f->uy));  CUDA_CHECK(cudaFree(f->chi_field));
    CUDA_CHECK(cudaFree(f->K_field));
}

void swap_populations(LBM_Populations* p1, LBM_Populations* p2) {
    LBM_Populations temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

// =========================================================
// KERNELS DE ACOPLAMENTO
// =========================================================

__device__ inline int get_idx(int x, int y) {
    return y * NX + x;
}

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = get_idx(x, y);
        double phi = fields.phi[idx];
        fields.chi_field[idx] = chi_max * 0.5 * (1.0 + phi);
    }
}

// =========================================================
// KERNEL LBM PRINCIPAL (BGK + FORÇAMENTO + STREAMING)
// =========================================================

__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < NX - 1 && y > 0 && y < NY - 1) {
        int idx = get_idx(x, y);

        // 1. Tensão de Korteweg
        double phi_c  = fields.phi[idx];
        double phi_R  = fields.phi[get_idx(x + 1, y)];
        double phi_L  = fields.phi[get_idx(x - 1, y)];
        double phi_T  = fields.phi[get_idx(x, y + 1)];
        double phi_B  = fields.phi[get_idx(x, y - 1)];

        double dx_phi = 0.5 * (phi_R - phi_L);
        double dy_phi = 0.5 * (phi_T - phi_B);
        double lap_phi = phi_R + phi_L + phi_T + phi_B - 4.0 * phi_c;

        double mu_c = 4.0 * BETA * phi_c * (phi_c * phi_c - 1.0) - KAPPA * lap_phi;
        double Fx = mu_c * dx_phi;
        double Fy = mu_c * dy_phi;

        // 2. Força Magnética de Kelvin
        double psi_c = fields.psi[idx];
        double psi_R = fields.psi[get_idx(x + 1, y)];
        double psi_L = fields.psi[get_idx(x - 1, y)];
        double psi_T = fields.psi[get_idx(x, y + 1)];
        double psi_B = fields.psi[get_idx(x, y - 1)];

        double psi_TR = fields.psi[get_idx(x + 1, y + 1)];
        double psi_TL = fields.psi[get_idx(x - 1, y + 1)];
        double psi_BR = fields.psi[get_idx(x + 1, y - 1)];
        double psi_BL = fields.psi[get_idx(x - 1, y - 1)];

        double hx = -0.5 * (psi_R - psi_L);
        double hy = -0.5 * (psi_T - psi_B);
        double d2psi_dx2 = psi_R - 2.0 * psi_c + psi_L;
        double d2psi_dy2 = psi_T - 2.0 * psi_c + psi_B;
        double d2psi_dxy = 0.25 * (psi_TR - psi_TL - psi_BR + psi_BL);

        double chi = fields.chi_field[idx];
        Fx += chi * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy));
        Fy += chi * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2));

        // 3. Termodinâmica e Momentos Locais
        double S_inv = fmax(0.0, fmin(1.0, (phi_c + 1.0) * 0.5));
        double tau = TAU_OUT + (TAU_IN - TAU_OUT) * S_inv;
        double omega = 1.0 / tau;
        double nu_local = (tau - 0.5) / 3.0;

        double k_local = fields.K_field[idx];
        double sigma_drag = nu_local / k_local;

        double f[9];
        f[0] = f_in.f0[idx]; f[1] = f_in.f1[idx]; f[2] = f_in.f2[idx];
        f[3] = f_in.f3[idx]; f[4] = f_in.f4[idx]; f[5] = f_in.f5[idx];
        f[6] = f_in.f6[idx]; f[7] = f_in.f7[idx]; f[8] = f_in.f8[idx];

        double rho_l = 0.0, ux_l = 0.0, uy_l = 0.0;
        for (int i = 0; i < 9; ++i) {
            rho_l += f[i];
            ux_l += f[i] * CX[i];
            uy_l += f[i] * CY[i];
        }

        // Esquema de Guo
        double ux_star = (ux_l + 0.5 * Fx) / rho_l;
        double uy_star = (uy_l + 0.5 * Fy) / rho_l;

        double ux_phys = ux_star / (1.0 + 0.5 * sigma_drag);
        double uy_phys = uy_star / (1.0 + 0.5 * sigma_drag);
        double usq = ux_phys * ux_phys + uy_phys * uy_phys;

        fields.rho[idx] = rho_l;
        fields.ux[idx]  = ux_phys;
        fields.uy[idx]  = uy_phys;

        double Fx_total = Fx - (sigma_drag * rho_l * ux_phys);
        double Fy_total = Fy - (sigma_drag * rho_l * uy_phys);

        // 4. Colisão BGK e Streaming
        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * ux_phys + CY[i] * uy_phys;
            double feq = W_LBM[i] * rho_l * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);

            double term1 = (CX[i] - ux_phys) * Fx_total + (CY[i] - uy_phys) * Fy_total;
            double term2 = cu * (CX[i] * Fx_total + CY[i] * Fy_total);
            double Si = W_LBM[i] * (1.0 - 0.5 * omega) * (3.0 * term1 + 9.0 * term2);

            double f_post = f[i] * (1.0 - omega) + omega * feq + Si;

            int be_x = x + CX[i];
            int be_y = y + CY[i];

            if (be_y >= 0 && be_y < NY) {
                if (be_x >= 0 && be_x < NX) {
                    int stream_idx = get_idx(be_x, be_y);
                    if(i==0) f_out.f0[stream_idx] = f_post;
                    else if(i==1) f_out.f1[stream_idx] = f_post;
                    else if(i==2) f_out.f2[stream_idx] = f_post;
                    else if(i==3) f_out.f3[stream_idx] = f_post;
                    else if(i==4) f_out.f4[stream_idx] = f_post;
                    else if(i==5) f_out.f5[stream_idx] = f_post;
                    else if(i==6) f_out.f6[stream_idx] = f_post;
                    else if(i==7) f_out.f7[stream_idx] = f_post;
                    else if(i==8) f_out.f8[stream_idx] = f_post;
                }
            } else {
                int opp = OPP[i];
                if(opp==0) f_out.f0[idx] = f_post;
                else if(opp==1) f_out.f1[idx] = f_post;
                else if(opp==2) f_out.f2[idx] = f_post;
                else if(opp==3) f_out.f3[idx] = f_post;
                else if(opp==4) f_out.f4[idx] = f_post;
                else if(opp==5) f_out.f5[idx] = f_post;
                else if(opp==6) f_out.f6[idx] = f_post;
                else if(opp==7) f_out.f7[idx] = f_post;
                else if(opp==8) f_out.f8[idx] = f_post;
            }
        }
    }
}

// =========================================================
// ESCOPO PRINCIPAL (HOST)
// =========================================================

int main() {
    LBM_Populations d_f_in, d_f_out;
    Macro_Fields d_fields;
    size_t mem_size = NUM_NODES * sizeof(double);

    // 1. Alocação de Memória GPU
    std::cout << "Alocando estruturas SoA na VRAM..." << std::endl;
    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    // 2. Alocação de Buffers Host
    std::cout << "Alocando Host Buffers na RAM..." << std::endl;
    double *h_phi = (double*)malloc(mem_size);
    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    if (!h_phi || !h_rho || !h_ux || !h_uy) {
        std::cerr << "Falha de segmentacao na RAM." << std::endl;
        exit(EXIT_FAILURE);
    }

    // 3. Inicialização de Sistema de Arquivos
    std::string output_dir = init_post_processing();
    std::cout << "Diretorio de saida configurado: " << output_dir << std::endl;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Resolvendo Problema de Valor Inicial..." << std::endl;
    int mode_m = 1;
    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields, mode_m);
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

    int max_iter = 2000;
    double chi_max = 1.2;

    std::cout << "Iniciando loop de integracao temporal LBM." << std::endl;

    for (int t = 0; t <= max_iter; ++t) {

        solve_cahn_hilliard(d_fields, numBlocks, threadsPerBlock);

        update_susceptibility_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, chi_max);
        CUDA_CHECK(cudaDeviceSynchronize());

        solve_poisson_magnetic(d_fields, numBlocks, threadsPerBlock);

        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        apply_open_boundaries(d_f_out, d_fields, NY);
        CUDA_CHECK(cudaDeviceSynchronize());

        swap_populations(&d_f_in, &d_f_out);

        if (t % SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_phi, h_rho, h_ux, h_uy);
            std::cout << "Snapshot " << t << " extraido." << std::endl;
        }
    }

    std::cout << "Liberando recursos (Garbage Collection)..." << std::endl;
    free(h_phi); free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out); free_macro_fields(&d_fields);

    return 0;
}