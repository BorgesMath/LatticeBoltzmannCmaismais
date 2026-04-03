// multiphase/cahn_hilliard.cu
#include "cahn_hilliard.cuh"
#include <cuda_runtime.h>
#include <algorithm> // Para std::swap, se usado no host

// =========================================================
// KERNEL 1: CÁLCULO DO POTENCIAL QUÍMICO (Passo 1)
// =========================================================
__global__ void compute_chemical_potential_kernel(Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        // Index Clamping para Condição de Neumann (Derivada Nula na fronteira)
        int xR = min(x + 1, NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, NY - 1);
        int yB = max(y - 1, 0);

        double phi_c = fields.phi[idx];
        double phi_R = fields.phi[y * NX + xR];
        double phi_L = fields.phi[y * NX + xL];
        double phi_T = fields.phi[yT * NX + x];
        double phi_B = fields.phi[yB * NX + x];

        // Laplaciano de Phi via diferenças finitas centrais de 2ª ordem
        double lap_phi = phi_R + phi_L + phi_T + phi_B - 4.0 * phi_c;

        // Cálculo da Tensão de Bulk (Poço Duplo) e Tensão Interfacial
        fields.mu[idx] = 4.0 * BETA * phi_c * (phi_c * phi_c - 1.0) - KAPPA * lap_phi;
    }
}

// =========================================================
// KERNEL 2: EVOLUÇÃO TEMPORAL - ADVECÇÃO/DIFUSÃO (Passo 2)
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

        // 1. Laplaciano do Potencial Químico
        double mu_c = fields.mu[idx];
        double mu_R = fields.mu[y * NX + xR];
        double mu_L = fields.mu[y * NX + xL];
        double mu_T = fields.mu[yT * NX + x];
        double mu_B = fields.mu[yB * NX + x];

        double lap_mu = mu_R + mu_L + mu_T + mu_B - 4.0 * mu_c;

        // 2. Termo de Advecção (u . grad(phi))
        double phi_R = fields.phi[y * NX + xR];
        double phi_L = fields.phi[y * NX + xL];
        double phi_T = fields.phi[yT * NX + x];
        double phi_B = fields.phi[yB * NX + x];

        double dx_phi = 0.5 * (phi_R - phi_L);
        double dy_phi = 0.5 * (phi_T - phi_B);

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

        // Dispara o cálculo de Mu
        compute_chemical_potential_kernel<<<numBlocks, threadsPerBlock>>>(fields);

        // Barreira de sincronização estrita (Garante que
        cudaDeviceSynchronize();

        // Dispara a evolução de Phi
        update_cahn_hilliard_kernel<<<numBlocks, threadsPerBlock>>>(fields);

        // Barreira de sincronização estrita
        cudaDeviceSynchronize();

        // Swap de Ponteiros Device-side
        // O campo phi_new se torna o campo atual para o próximo sub-passo,
        // e o array antigo de phi servirá como rascunho na próxima iteração.
        double* temp = fields.phi;
        fields.phi = fields.phi_new;
        fields.phi_new = temp;
    }
}