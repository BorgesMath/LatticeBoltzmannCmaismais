#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, LBM_Populations_Phase g_in, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = y * cfg.NX + x;

        double x_center = (double)cfg.NX * 0.40;
        double k_wave = 2.0 * PI * cfg.MODE_M / (double)cfg.NY;
        double dist = x_center + cfg.INITIAL_AMPLITUDE * cos(k_wave * y);

        double phi_val = -tanh((x - dist) / (cfg.INTERFACE_WIDTH / 2.0));
        fields.phi[idx] = phi_val;

        fields.K_field[idx] = cfg.K_0;
        fields.rho[idx] = 1.0;
        fields.ux[idx]  = cfg.U_INLET;
        fields.uy[idx]  = 0.0;

        double angle_rad = cfg.H_ANGLE * PI / 180.0;
        fields.psi[idx] = (cfg.H0 * cos(angle_rad)) * (cfg.NX - x) + (cfg.H0 * sin(angle_rad)) * (cfg.NY - y);

        double u_sq = cfg.U_INLET * cfg.U_INLET;

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * cfg.U_INLET;

            double feq = W_LBM[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            // Equilíbrio conservativo limpo
            double geq = W_LBM[i] * phi_val * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

            if (i == 0) { f_in.f0[idx] = feq; g_in.g0[idx] = geq; }
            else if (i == 1) { f_in.f1[idx] = feq; g_in.g1[idx] = geq; }
            else if (i == 2) { f_in.f2[idx] = feq; g_in.g2[idx] = geq; }
            else if (i == 3) { f_in.f3[idx] = feq; g_in.g3[idx] = geq; }
            else if (i == 4) { f_in.f4[idx] = feq; g_in.g4[idx] = geq; }
            else if (i == 5) { f_in.f5[idx] = feq; g_in.g5[idx] = geq; }
            else if (i == 6) { f_in.f6[idx] = feq; g_in.g6[idx] = geq; }
            else if (i == 7) { f_in.f7[idx] = feq; g_in.g7[idx] = geq; }
            else if (i == 8) { f_in.f8[idx] = feq; g_in.g8[idx] = geq; }
        }
    }
}