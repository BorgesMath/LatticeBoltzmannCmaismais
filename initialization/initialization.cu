#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = y * cfg.NX + x;

        double angle_rad = cfg.H_ANGLE * PI / 180.0;
        double Hx = cfg.H0 * cos(angle_rad);
        double Hy = cfg.H0 * sin(angle_rad);

        double x_center = (double)cfg.NX * 0.40;
        double dist = x_center + cfg.INITIAL_AMPLITUDE * cos(2.0 * PI * cfg.MODE_M * y / (double)cfg.NY);

        fields.phi[idx] = -tanh((x - dist) / (cfg.INTERFACE_WIDTH / 2.0));

        fields.K_field[idx] = cfg.K_0;
        fields.rho[idx] = 1.0;

        fields.ux[idx]  = cfg.U_INLET;
        fields.uy[idx]  = 0.0;
        fields.psi[idx] = Hx * (cfg.NX - x) + Hy * (cfg.NY - y);

        double u_sq = cfg.U_INLET * cfg.U_INLET;
        double rho_local = 1.0;

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * cfg.U_INLET;
            double feq = W_LBM[i] * rho_local * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

            if (i == 0) f_in.f0[idx] = feq;
            else if (i == 1) f_in.f1[idx] = feq;
            else if (i == 2) f_in.f2[idx] = feq;
            else if (i == 3) f_in.f3[idx] = feq;
            else if (i == 4) f_in.f4[idx] = feq;
            else if (i == 5) f_in.f5[idx] = feq;
            else if (i == 6) f_in.f6[idx] = feq;
            else if (i == 7) f_in.f7[idx] = feq;
            else if (i == 8) f_in.f8[idx] = feq;
        }
    }
}