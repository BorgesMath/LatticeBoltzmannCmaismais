// Magnetismo/poisson.cu
#include "poisson.cuh"
#include <cmath>

// =========================================================
// KERNEL 1: RED-BLACK SOR (Operador Elíptico)
// =========================================================
__global__ void poisson_red_black_kernel(Macro_Fields fields, int color) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Garante que opera apenas no interior (preservando Dirichlet em x=0 e x=NX-1)
    if (x > 0 && x < NX - 1 && y > 0 && y < NY - 1) {

        // Aplicação do filtro de paridade (0 para Vermelho, 1 para Preto)
        if ((x + y) % 2 == color) {
            int idx = y * NX + x;

            // Interpolação da permeabilidade magnética (mu) nas faces
            double chi_c = fields.chi_field[idx];
            double chi_E = fields.chi_field[y * NX + (x + 1)];
            double chi_W = fields.chi_field[y * NX + (x - 1)];
            double chi_N = fields.chi_field[(y + 1) * NX + x];
            double chi_S = fields.chi_field[(y - 1) * NX + x];

            double mu_E = 1.0 + 0.5 * (chi_c + chi_E);
            double mu_W = 1.0 + 0.5 * (chi_c + chi_W);
            double mu_N = 1.0 + 0.5 * (chi_c + chi_N);
            double mu_S = 1.0 + 0.5 * (chi_c + chi_S);

            double denom = mu_E + mu_W + mu_N + mu_S;

            if (denom > 1e-12) {
                double psi_E = fields.psi[y * NX + (x + 1)];
                double psi_W = fields.psi[y * NX + (x - 1)];
                double psi_N = fields.psi[(y + 1) * NX + x];
                double psi_S = fields.psi[(y - 1) * NX + x];

                double psi_new = (mu_E * psi_E + mu_W * psi_W +
                                  mu_N * psi_N + mu_S * psi_S) / denom;

                // Relaxação SOR
                fields.psi[idx] = (1.0 - SOR_OMEGA) * fields.psi[idx] + SOR_OMEGA * psi_new;
            }
        }
    }
}

// =========================================================
// KERNEL 2: CONDIÇÕES DE CONTORNO DE NEUMANN (Paredes Transversais)
// =========================================================
__global__ void apply_magnetic_neumann_bc(Macro_Fields fields, double Hy_target) {
    // Topologia 1D: Uma thread por coluna X
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < NX) {
        // Fundo (y = 0)
        fields.psi[x] = fields.psi[1 * NX + x] + Hy_target;

        // Topo (y = NY - 1)
        fields.psi[(NY - 1) * NX + x] = fields.psi[(NY - 2) * NX + x] - Hy_target;
    }
}

// =========================================================
// ORQUESTRADOR HOST: GERENCIA O LOOP SOR
// =========================================================
void solve_poisson_magnetic(Macro_Fields fields, dim3 numBlocks2D, dim3 threadsPerBlock2D) {

    // Pré-cálculo da componente vetorial na CPU
    double angle_rad = H_ANGLE * PI / 180.0;
    double Hy_target = H0 * sin(angle_rad);

    // Configuração de Topologia 1D para o Kernel de Fronteira
    int threads1D = 256;
    int blocks1D = (NX + threads1D - 1) / threads1D;

    for (int i = 0; i < SOR_ITERATIONS; ++i) {

        // Passo 1: Atualiza nós Vermelhos (color = 0)
        poisson_red_black_kernel<<<numBlocks2D, threadsPerBlock2D>>>(fields, 0);
        cudaDeviceSynchronize();

        // Passo 2: Atualiza nós Pretos (color = 1)
        poisson_red_black_kernel<<<numBlocks2D, threadsPerBlock2D>>>(fields, 1);
        cudaDeviceSynchronize();

        // Passo 3: Imposição forçada de Neumann nas fronteiras
        apply_magnetic_neumann_bc<<<blocks1D, threads1D>>>(fields, Hy_target);
        cudaDeviceSynchronize();
    }
}