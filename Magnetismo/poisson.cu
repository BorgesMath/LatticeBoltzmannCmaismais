#include "poisson.cuh"
#include <cmath>

__global__ void poisson_red_black_kernel(Macro_Fields fields, int color, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < cfg.NX - 1 && y > 0 && y < cfg.NY - 1) {
        if ((x + y) % 2 == color) {
            int idx = y * cfg.NX + x;

            double chi_c = fields.chi_field[idx];
            double chi_E = fields.chi_field[y * cfg.NX + (x + 1)];
            double chi_W = fields.chi_field[y * cfg.NX + (x - 1)];
            double chi_N = fields.chi_field[(y + 1) * cfg.NX + x];
            double chi_S = fields.chi_field[(y - 1) * cfg.NX + x];

            double mu_E = 1.0 + 0.5 * (chi_c + chi_E);
            double mu_W = 1.0 + 0.5 * (chi_c + chi_W);
            double mu_N = 1.0 + 0.5 * (chi_c + chi_N);
            double mu_S = 1.0 + 0.5 * (chi_c + chi_S);

            double denom = mu_E + mu_W + mu_N + mu_S;

            if (denom > 1e-12) {
                double psi_E = fields.psi[y * cfg.NX + (x + 1)];
                double psi_W = fields.psi[y * cfg.NX + (x - 1)];
                double psi_N = fields.psi[(y + 1) * cfg.NX + x];
                double psi_S = fields.psi[(y - 1) * cfg.NX + x];

                double psi_new = (mu_E * psi_E + mu_W * psi_W +
                                  mu_N * psi_N + mu_S * psi_S) / denom;

                fields.psi[idx] = (1.0 - cfg.SOR_OMEGA) * fields.psi[idx] + cfg.SOR_OMEGA * psi_new;
            }
        }
    }
}

__global__ void apply_magnetic_neumann_bc(Macro_Fields fields, double Hy_target, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < cfg.NX) {
        fields.psi[x] = fields.psi[1 * cfg.NX + x] + Hy_target;
        fields.psi[(cfg.NY - 1) * cfg.NX + x] = fields.psi[(cfg.NY - 2) * cfg.NX + x] - Hy_target;
    }
}

void solve_poisson_magnetic(Macro_Fields fields, dim3 numBlocks2D, dim3 threadsPerBlock2D, SimConfig cfg) {
    double angle_rad = cfg.H_ANGLE * PI / 180.0;
    double Hy_target = cfg.H0 * sin(angle_rad);

    int threads1D = 256;
    int blocks1D = (cfg.NX + threads1D - 1) / threads1D;

    for (int i = 0; i < cfg.SOR_ITERATIONS; ++i) {
        poisson_red_black_kernel<<<numBlocks2D, threadsPerBlock2D>>>(fields, 0, cfg);
        poisson_red_black_kernel<<<numBlocks2D, threadsPerBlock2D>>>(fields, 1, cfg);
        apply_magnetic_neumann_bc<<<blocks1D, threads1D>>>(fields, Hy_target, cfg);
    }
    // Zero invocações de cudaDeviceSynchronize(). O fluxo é mantido na VRAM em alta taxa.
}