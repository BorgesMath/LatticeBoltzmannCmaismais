#include "open_boundaries.cuh"
#include <cuda_runtime.h>

__global__ void open_boundaries_kernel(LBM_Populations f_out, Macro_Fields fields, double U_in_val) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < NY) {
        // =========================================================
        // 1. OUTLET (x = NX - 1) -> Dirichlet de Pressão (rho = 1.0)
        // =========================================================
        int idx_out = y * NX + (NX - 1);
        int idx_out_ref = y * NX + (NX - 2);

        // Ancoragem termodinâmica rigorosa: Evita drift de densidade (NaN)
        double rho_out = 1.0;
        double ux_out = fields.ux[idx_out_ref]; // Velocidade desenvolvida
        double uy_out = fields.uy[idx_out_ref];
        double u_sq_out = ux_out * ux_out + uy_out * uy_out;

        // =========================================================
        // 2. INLET (x = 0) -> Dirichlet em Velocidade
        // =========================================================
        int idx_in = y * NX + 0;
        int idx_in_ref = y * NX + 1;

        double rho_inlet = fields.rho[idx_in_ref];
        double u_sq_in = U_in_val * U_in_val;

        for (int i = 0; i < 9; ++i) {
            // Reconstrução de Equilíbrio no Outlet
            double cu_out = CX[i] * ux_out + CY[i] * uy_out;
            double feq_out = W_LBM[i] * rho_out * (1.0 + 3.0 * cu_out + 4.5 * cu_out * cu_out - 1.5 * u_sq_out);

            // Reconstrução de Equilíbrio no Inlet
            double cu_in = CX[i] * U_in_val;
            double feq_in = W_LBM[i] * rho_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in * cu_in - 1.5 * u_sq_in);

            if (i == 0) { f_out.f0[idx_out] = feq_out; f_out.f0[idx_in] = feq_in; }
            else if (i == 1) { f_out.f1[idx_out] = feq_out; f_out.f1[idx_in] = feq_in; }
            else if (i == 2) { f_out.f2[idx_out] = feq_out; f_out.f2[idx_in] = feq_in; }
            else if (i == 3) { f_out.f3[idx_out] = feq_out; f_out.f3[idx_in] = feq_in; }
            else if (i == 4) { f_out.f4[idx_out] = feq_out; f_out.f4[idx_in] = feq_in; }
            else if (i == 5) { f_out.f5[idx_out] = feq_out; f_out.f5[idx_in] = feq_in; }
            else if (i == 6) { f_out.f6[idx_out] = feq_out; f_out.f6[idx_in] = feq_in; }
            else if (i == 7) { f_out.f7[idx_out] = feq_out; f_out.f7[idx_in] = feq_in; }
            else if (i == 8) { f_out.f8[idx_out] = feq_out; f_out.f8[idx_in] = feq_in; }
        }
    }
}

void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim, double U_in_val) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (NY_dim + threadsPerBlock - 1) / threadsPerBlock;
    open_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_f_out, d_fields, U_in_val);
}