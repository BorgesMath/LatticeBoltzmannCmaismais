#include "cahn_hilliard.cuh"
#include <cuda_runtime.h>
#include <algorithm>

__global__ void compute_chemical_potential_kernel(Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = y * cfg.NX + x;

        int xR = min(x + 1, cfg.NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, cfg.NY - 1);
        int yB = max(y - 1, 0);

        double phi_c  = fields.phi[idx];
        double phi_R  = fields.phi[y * cfg.NX + xR];
        double phi_L  = fields.phi[y * cfg.NX + xL];
        double phi_T  = fields.phi[yT * cfg.NX + x];
        double phi_B  = fields.phi[yB * cfg.NX + x];
        double phi_TR = fields.phi[yT * cfg.NX + xR];
        double phi_TL = fields.phi[yT * cfg.NX + xL];
        double phi_BR = fields.phi[yB * cfg.NX + xR];
        double phi_BL = fields.phi[yB * cfg.NX + xL];

        double lap_phi = (1.0 / 6.0) * (4.0 * (phi_R + phi_L + phi_T + phi_B) +
                                        1.0 * (phi_TR + phi_TL + phi_BR + phi_BL) - 20.0 * phi_c);

        fields.mu[idx] = 4.0 * cfg.BETA * phi_c * (phi_c * phi_c - 1.0) - cfg.KAPPA * lap_phi;
    }
}

__global__ void update_cahn_hilliard_kernel(Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = y * cfg.NX + x;

        int xR = min(x + 1, cfg.NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, cfg.NY - 1);
        int yB = max(y - 1, 0);

        double mu_c  = fields.mu[idx];
        double mu_R  = fields.mu[y * cfg.NX + xR];
        double mu_L  = fields.mu[y * cfg.NX + xL];
        double mu_T  = fields.mu[yT * cfg.NX + x];
        double mu_B  = fields.mu[yB * cfg.NX + x];
        double mu_TR = fields.mu[yT * cfg.NX + xR];
        double mu_TL = fields.mu[yT * cfg.NX + xL];
        double mu_BR = fields.mu[yB * cfg.NX + xR];
        double mu_BL = fields.mu[yB * cfg.NX + xL];

        double lap_mu = (1.0 / 6.0) * (4.0 * (mu_R + mu_L + mu_T + mu_B) +
                                       1.0 * (mu_TR + mu_TL + mu_BR + mu_BL) - 20.0 * mu_c);

        double phi_R  = fields.phi[y * cfg.NX + xR];
        double phi_L  = fields.phi[y * cfg.NX + xL];
        double phi_T  = fields.phi[yT * cfg.NX + x];
        double phi_B  = fields.phi[yB * cfg.NX + x];
        double phi_TR = fields.phi[yT * cfg.NX + xR];
        double phi_TL = fields.phi[yT * cfg.NX + xL];
        double phi_BR = fields.phi[yB * cfg.NX + xR];
        double phi_BL = fields.phi[yB * cfg.NX + xL];

        double dx_phi = (1.0 / 6.0) * (2.0 * (phi_R - phi_L) + (phi_TR + phi_BR) - (phi_TL + phi_BL));
        double dy_phi = (1.0 / 6.0) * (2.0 * (phi_T - phi_B) + (phi_TR + phi_TL) - (phi_BR + phi_BL));

        double ux = fields.ux[idx];
        double uy = fields.uy[idx];

        double advection = ux * dx_phi + uy * dy_phi;

        fields.phi_new[idx] = fields.phi[idx] + cfg.DT_CH * (cfg.M_MOBILITY * lap_mu - advection);
    }
}

void solve_cahn_hilliard(Macro_Fields& fields, dim3 numBlocks, dim3 threadsPerBlock, SimConfig cfg) {
    for (int step = 0; step < cfg.CH_SUBSTEPS; ++step) {
        // Kernels enfileirados assincronamente (Stream 0)
        compute_chemical_potential_kernel<<<numBlocks, threadsPerBlock>>>(fields, cfg);
        update_cahn_hilliard_kernel<<<numBlocks, threadsPerBlock>>>(fields, cfg);

        // Swap em O(1) na CPU. Reflete nos próximos kernels pois fields é passado por referência.
        double* temp = fields.phi;
        fields.phi = fields.phi_new;
        fields.phi_new = temp;
    }
}