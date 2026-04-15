#include "open_boundaries.cuh"
#include <cuda_runtime.h>

__global__ void open_boundaries_kernel(LBM_Populations f_out, Macro_Fields fields, double U_in_val) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < NY) {
        // ... [MANTENHA A PARTE DO OUTLET INTACTA] ...

        // =========================================================
        // 2. INLET (x = 0) -> Dirichlet em Velocidade via Equilíbrio
        // =========================================================
        int idx_in = y * NX + 0;
        int idx_in_ref = y * NX + 1;

        double rho_inlet = fields.rho[idx_in_ref];
        double u_sq = U_in_val * U_in_val; // <--- Uso da variável injetada

        for (int i = 0; i < 9; ++i) {
            double cu_in = CX[i] * U_in_val; // <--- Uso da variável injetada
            double feq_in = W_LBM[i] * rho_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in * cu_in - 1.5 * u_sq);

            if (i == 0) f_out.f0[idx_in] = feq_in;
            else if (i == 1) f_out.f1[idx_in] = feq_in;
            // ... [Mantenha os outros if-elses intactos]
            else if (i == 8) f_out.f8[idx_in] = feq_in;
        }
    }
}

void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim, double U_in_val) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (NY_dim + threadsPerBlock - 1) / threadsPerBlock;
    open_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_f_out, d_fields, U_in_val);
}