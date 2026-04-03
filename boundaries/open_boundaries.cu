// boundaries/open_boundaries.cu
#include "open_boundaries.cuh"
#include <cuda_runtime.h>

__global__ void open_boundaries_kernel(LBM_Populations f_out, Macro_Fields fields) {
    // Topologia 1D (percorre apenas o eixo Y)
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < NY) {

        // =========================================================
        // 1. OUTLET (x = NX - 1) -> Extrapolação Convectiva
        // =========================================================
        int idx_out   = y * NX + (NX - 1);
        int idx_out_1 = y * NX + (NX - 2);
        int idx_out_2 = y * NX + (NX - 3);

        f_out.f0[idx_out] = 2.0 * f_out.f0[idx_out_1] - f_out.f0[idx_out_2];
        f_out.f1[idx_out] = 2.0 * f_out.f1[idx_out_1] - f_out.f1[idx_out_2];
        f_out.f2[idx_out] = 2.0 * f_out.f2[idx_out_1] - f_out.f2[idx_out_2];
        f_out.f3[idx_out] = 2.0 * f_out.f3[idx_out_1] - f_out.f3[idx_out_2];
        f_out.f4[idx_out] = 2.0 * f_out.f4[idx_out_1] - f_out.f4[idx_out_2];
        f_out.f5[idx_out] = 2.0 * f_out.f5[idx_out_1] - f_out.f5[idx_out_2];
        f_out.f6[idx_out] = 2.0 * f_out.f6[idx_out_1] - f_out.f6[idx_out_2];
        f_out.f7[idx_out] = 2.0 * f_out.f7[idx_out_1] - f_out.f7[idx_out_2];
        f_out.f8[idx_out] = 2.0 * f_out.f8[idx_out_1] - f_out.f8[idx_out_2];

        // =========================================================
        // 2. INLET (x = 0) -> Dirichlet em Velocidade via Equilíbrio
        // =========================================================
        int idx_in = y * NX + 0;
        int idx_in_ref = y * NX + 1; // Nó interior imediato para capturar o gradiente de pressão

        // Extrapola a densidade do nó interior para permitir a formação natural do campo de Darcy
        double rho_inlet = fields.rho[idx_in_ref];

        double u_sq = U_INLET * U_INLET;

        // Forçamento do estado populacional para u = [U_INLET, 0]
        for (int i = 0; i < 9; ++i) {
            double cu_in = CX[i] * U_INLET; // O vetor CY é 0 para o Inlet horizontal
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

// Orquestrador Host
void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (NY_dim + threadsPerBlock - 1) / threadsPerBlock;

    open_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_f_out, d_fields);
    // Sincronização dispensada temporariamente aqui se o próximo passo não for leitura síncrona
}