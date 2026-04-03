#include "cahn_hilliard.cuh"
#include <cuda_runtime.h>
#include <algorithm>

// =========================================================
// KERNEL 1: CÁLCULO DO POTENCIAL QUÍMICO (Laplaciano Isotrópico)
// =========================================================
__global__ void compute_chemical_potential_kernel(Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        // Index Clamping para Neumann nas Diagonais e Ortogonais
        int xR = min(x + 1, NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, NY - 1);
        int yB = max(y - 1, 0);

        double phi_c  = fields.phi[idx];
        double phi_R  = fields.phi[y * NX + xR];
        double phi_L  = fields.phi[y * NX + xL];
        double phi_T  = fields.phi[yT * NX + x];
        double phi_B  = fields.phi[yB * NX + x];
        double phi_TR = fields.phi[yT * NX + xR];
        double phi_TL = fields.phi[yT * NX + xL];
        double phi_BR = fields.phi[yB * NX + xR];
        double phi_BL = fields.phi[yB * NX + xL];

        // Laplaciano Isotrópico (D2Q9 Estrito)
        double lap_phi = (1.0 / 6.0) * (4.0 * (phi_R + phi_L + phi_T + phi_B) +
                                        1.0 * (phi_TR + phi_TL + phi_BR + phi_BL) - 20.0 * phi_c);

        fields.mu[idx] = 4.0 * BETA * phi_c * (phi_c * phi_c - 1.0) - KAPPA * lap_phi;
    }
}

// =========================================================
// KERNEL 2: EVOLUÇÃO TEMPORAL (Gradiente e Laplaciano Isotrópicos)
// =========================================================
__global__ void update_cahn_hilliard_kernel(Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        int xR = min(x + 1, NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, NY - 1);
        int yB = max(y - 1, 0);

        // 1. Laplaciano Isotrópico do Potencial Químico (Mu)
        double mu_c  = fields.mu[idx];
        double mu_R  = fields.mu[y * NX + xR];
        double mu_L  = fields.mu[y * NX + xL];
        double mu_T  = fields.mu[yT * NX + x];
        double mu_B  = fields.mu[yB * NX + x];
        double mu_TR = fields.mu[yT * NX + xR];
        double mu_TL = fields.mu[yT * NX + xL];
        double mu_BR = fields.mu[yB * NX + xR];
        double mu_BL = fields.mu[yB * NX + xL];

        double lap_mu = (1.0 / 6.0) * (4.0 * (mu_R + mu_L + mu_T + mu_B) +
                                       1.0 * (mu_TR + mu_TL + mu_BR + mu_BL) - 20.0 * mu_c);

        // 2. Gradiente Isotrópico do Parâmetro de Ordem (Phi) para Advecção
        double phi_R  = fields.phi[y * NX + xR];
        double phi_L  = fields.phi[y * NX + xL];
        double phi_T  = fields.phi[yT * NX + x];
        double phi_B  = fields.phi[yB * NX + x];
        double phi_TR = fields.phi[yT * NX + xR];
        double phi_TL = fields.phi[yT * NX + xL];
        double phi_BR = fields.phi[yB * NX + xR];
        double phi_BL = fields.phi[yB * NX + xL];

        double dx_phi = (1.0 / 6.0) * (2.0 * (phi_R - phi_L) + (phi_TR + phi_BR) - (phi_TL + phi_BL));
        double dy_phi = (1.0 / 6.0) * (2.0 * (phi_T - phi_B) + (phi_TR + phi_TL) - (phi_BR + phi_BL));

        double ux = fields.ux[idx];
        double uy = fields.uy[idx];

        double advection = ux * dx_phi + uy * dy_phi;

        // 3. Integração de Euler Explícito
        fields.phi_new[idx] = fields.phi[idx] + DT_CH * (M_MOBILITY * lap_mu - advection);
    }
}

// =========================================================
// ORQUESTRADOR HOST: GERENCIA O SUB-STEPPING
// =========================================================
void solve_cahn_hilliard(Macro_Fields fields, dim3 numBlocks, dim3 threadsPerBlock) {
    for (int step = 0; step < CH_SUBSTEPS; ++step) {
        compute_chemical_potential_kernel<<<numBlocks, threadsPerBlock>>>(fields);
        cudaDeviceSynchronize();

        update_cahn_hilliard_kernel<<<numBlocks, threadsPerBlock>>>(fields);
        cudaDeviceSynchronize();

        // Swap Device-side
        double* temp = fields.phi;
        fields.phi = fields.phi_new;
        fields.phi_new = temp;
    }
}