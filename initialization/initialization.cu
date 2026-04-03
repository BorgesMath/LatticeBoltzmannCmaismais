// initialization/initialization.cu
#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields, int mode_m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        double angle_rad = H_ANGLE * PI / 180.0;
        double Hx = H0 * cos(angle_rad);
        double Hy = H0 * sin(angle_rad);

        double x_center = NX * (80.0 / 500.0) + 600.0;
        double dist = x_center + INITIAL_AMPLITUDE * cos(2.0 * PI * mode_m * y / (double)NY);

        fields.phi[idx] = -tanh((x - dist) / (INTERFACE_WIDTH / 2.0));
        fields.K_field[idx] = K_0;
        fields.rho[idx] = 1.0;
        fields.ux[idx]  = 0.0;
        fields.uy[idx]  = 0.0;
        fields.psi[idx] = Hx * (NX - x) + Hy * (NY - y);

        f_in.f0[idx] = W_LBM[0] * 1.0;
        f_in.f1[idx] = W_LBM[1] * 1.0;
        f_in.f2[idx] = W_LBM[2] * 1.0;
        f_in.f3[idx] = W_LBM[3] * 1.0;
        f_in.f4[idx] = W_LBM[4] * 1.0;
        f_in.f5[idx] = W_LBM[5] * 1.0;
        f_in.f6[idx] = W_LBM[6] * 1.0;
        f_in.f7[idx] = W_LBM[7] * 1.0;
        f_in.f8[idx] = W_LBM[8] * 1.0;
    }
}