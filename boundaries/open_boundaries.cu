#include "open_boundaries.cuh"
#include <cuda_runtime.h>

__global__ void open_boundaries_kernel(LBM_Populations f_out, LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < cfg.NY) {
        // 1. OUTLET (Saída Direita) - Extrapolação de 2ª Ordem
        int idx_out   = y * cfg.NX + (cfg.NX - 1);
        int idx_out_1 = y * cfg.NX + (cfg.NX - 2);
        int idx_out_2 = y * cfg.NX + (cfg.NX - 3);

        // Hidrodinâmica
        f_out.f0[idx_out] = 2.0 * f_out.f0[idx_out_1] - f_out.f0[idx_out_2];
        f_out.f1[idx_out] = 2.0 * f_out.f1[idx_out_1] - f_out.f1[idx_out_2];
        f_out.f2[idx_out] = 2.0 * f_out.f2[idx_out_1] - f_out.f2[idx_out_2];
        f_out.f3[idx_out] = 2.0 * f_out.f3[idx_out_1] - f_out.f3[idx_out_2];
        f_out.f4[idx_out] = 2.0 * f_out.f4[idx_out_1] - f_out.f4[idx_out_2];
        f_out.f5[idx_out] = 2.0 * f_out.f5[idx_out_1] - f_out.f5[idx_out_2];
        f_out.f6[idx_out] = 2.0 * f_out.f6[idx_out_1] - f_out.f6[idx_out_2];
        f_out.f7[idx_out] = 2.0 * f_out.f7[idx_out_1] - f_out.f7[idx_out_2];
        f_out.f8[idx_out] = 2.0 * f_out.f8[idx_out_1] - f_out.f8[idx_out_2];

        // Campo de fase: cópia de 1ª ordem
        g_out.g0[idx_out] = g_out.g0[idx_out_1];
        g_out.g1[idx_out] = g_out.g1[idx_out_1];
        g_out.g2[idx_out] = g_out.g2[idx_out_1];
        g_out.g3[idx_out] = g_out.g3[idx_out_1];
        g_out.g4[idx_out] = g_out.g4[idx_out_1];
        g_out.g5[idx_out] = g_out.g5[idx_out_1];
        g_out.g6[idx_out] = g_out.g6[idx_out_1];
        g_out.g7[idx_out] = g_out.g7[idx_out_1];
        g_out.g8[idx_out] = g_out.g8[idx_out_1];


        // 2. INLET (Entrada Esquerda) - Dirichlet Constante
        int idx_in = y * cfg.NX + 0;
        int idx_in_ref = y * cfg.NX + 1;

        double rho_inlet = fields.rho[idx_in_ref];
        double u_sq = cfg.U_INLET * cfg.U_INLET;

        // Em um deslocamento, a face de entrada injeta perpetuamente o fluido deslocador (phi = -1.0)
        double phi_inlet = 1.0;

        for (int i = 0; i < 9; ++i) {
            double cu_in = CX[i] * cfg.U_INLET;

            double feq_in = W_LBM[i] * rho_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in * cu_in - 1.5 * u_sq);
            double geq_in = W_LBM[i] * phi_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in * cu_in - 1.5 * u_sq);

            if (i == 0) { f_out.f0[idx_in] = feq_in; g_out.g0[idx_in] = geq_in; }
            else if (i == 1) { f_out.f1[idx_in] = feq_in; g_out.g1[idx_in] = geq_in; }
            else if (i == 2) { f_out.f2[idx_in] = feq_in; g_out.g2[idx_in] = geq_in; }
            else if (i == 3) { f_out.f3[idx_in] = feq_in; g_out.g3[idx_in] = geq_in; }
            else if (i == 4) { f_out.f4[idx_in] = feq_in; g_out.g4[idx_in] = geq_in; }
            else if (i == 5) { f_out.f5[idx_in] = feq_in; g_out.g5[idx_in] = geq_in; }
            else if (i == 6) { f_out.f6[idx_in] = feq_in; g_out.g6[idx_in] = geq_in; }
            else if (i == 7) { f_out.f7[idx_in] = feq_in; g_out.g7[idx_in] = geq_in; }
            else if (i == 8) { f_out.f8[idx_in] = feq_in; g_out.g8[idx_in] = geq_in; }
        }
    }
}

void apply_open_boundaries(LBM_Populations d_f_out, LBM_Populations_Phase d_g_out, Macro_Fields d_fields, int NY_dim, SimConfig cfg) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (NY_dim + threadsPerBlock - 1) / threadsPerBlock;
    open_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_f_out, d_g_out, d_fields, cfg);
}