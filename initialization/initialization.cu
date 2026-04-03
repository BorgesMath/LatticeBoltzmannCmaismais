#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        // Geometria da Gota Estática
        double xc = NX / 2.0;
        double yc = NY / 2.0;
        double R0 = 20.0;

        double dist = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));

        // Espessura numérica extraída diretamente da álgebra configurada
        double W_interface = sqrt(KAPPA / (2.0 * BETA));

        // Injeção do perfil de equilíbrio termodinâmico de energia mínima
        fields.phi[idx] = -tanh((dist - R0) / W_interface);









        fields.K_field[idx] = K_0;
        fields.rho[idx] = 1.0;
        fields.ux[idx]  = 0.0;
        fields.uy[idx]  = 0.0;
        fields.psi[idx] = 0.0;

        // Inicialização do Equilíbrio
        f_in.f0[idx] = W_LBM[0];
        f_in.f1[idx] = W_LBM[1]; f_in.f2[idx] = W_LBM[2];
        f_in.f3[idx] = W_LBM[3]; f_in.f4[idx] = W_LBM[4];
        f_in.f5[idx] = W_LBM[5]; f_in.f6[idx] = W_LBM[6];
        f_in.f7[idx] = W_LBM[7]; f_in.f8[idx] = W_LBM[8];
    }
}