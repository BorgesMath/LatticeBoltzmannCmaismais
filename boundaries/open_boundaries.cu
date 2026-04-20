#include "open_boundaries.cuh"
#include <cuda_runtime.h>

__global__ void open_boundaries_kernel(LBM_Populations f_out, Macro_Fields fields, SimConfig cfg) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < cfg.NY) {
        int idx_out   = y * cfg.NX + (cfg.NX - 1);
        int idx_out_1 = y * cfg.NX + (cfg.NX - 2);
        int idx_out_2 = y * cfg.NX + (cfg.NX - 3);

        f_out.f0[idx_out] = 2.0 * f_out.f0[idx_out_1] - f_out.f0[idx_out_2];
        f_out.f1[idx_out] = 2.0 * f_out.f1[idx_out_1] - f_out.f1[idx_out_2];
        f_out.f2[idx_out] = 2.0 * f_out.f2[idx_out_1] - f_out.f2[idx_out_2];
        f_out.f3[idx_out] = 2.0 * f_out.f3[idx_out_1] - f_out.f3[idx_out_2];
        f_out.f4[idx_out] = 2.0 * f_out.f4[idx_out_1] - f_out.f4[idx_out_2];
        f_out.f5[idx_out] = 2.0 * f_out.f5[idx_out_1] - f_out.f5[idx_out_2];
        f_out.f6[idx_out] = 2.0 * f_out.f6[idx_out_1] - f_out.f6[idx_out_2];
        f_out.f7[idx_out] = 2.0 * f_out.f7[idx_out_1] - f_out.f7[idx_out_2];
        f_out.f8[idx_out] = 2.0 * f_out.f8[idx_out_1] - f_out.f8[idx_out_2];

        int idx_in = y * cfg.NX + 0;
        int idx_in_ref = y * cfg.NX + 1;

        double rho_inlet = fields.rho[idx_in_ref];
        double u_sq = cfg.U_INLET * cfg.U_INLET;

        for (int i = 0; i < 9; ++i) {
            double cu_in = CX[i] * cfg.U_INLET;
            double feq_in = W_LBM[i] * rho_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in * cu_in - 1.5 * u_sq);

            if (i == 0) f_out.f0[idx_in] = feq_in;
            else if (i == 1) f_out.f1[idx_in] = feq_in;
            else if (i == 2) f_out.f2[idx_in] = feq_in;
            else if (i == 3) f_out.f3[idx_in] = feq_in;
            else if (i == 4) f_out.f4[idx_in] = feq_in;
            else if (i == 5) f_out.f5[idx_in] = feq_in;
            else if (i == 6) f_out.f6[idx_in] = feq_in;
            else if (i == 7) f_out.f7[idx_in] = feq_in;
            else if (i == 8) f_out.f8[idx_in] = feq_in;
        }
    }
}

void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim, SimConfig cfg) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (NY_dim + threadsPerBlock - 1) / threadsPerBlock;
    open_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_f_out, d_fields, cfg);
}