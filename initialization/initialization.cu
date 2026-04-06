// initialization/initialization.cu
#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = y * NX + x;

        // Projeção do Campo Magnético
        double angle_rad = H_ANGLE * PI / 180.0;
        double Hx = H0 * cos(angle_rad);
        double Hy = H0 * sin(angle_rad);

        // Perturbação de Interface Saffman-Taylor
        double x_center = (double)NX * 0.40;
        double dist = x_center + INITIAL_AMPLITUDE * cos(2.0 * PI * MODE_M * y / (double)NY);

        fields.phi[idx] = -tanh((x - dist) / (INTERFACE_WIDTH / 2.0));

        // Constantes reológicas e momentos de ordem zero e um
        fields.K_field[idx] = K_0;
        fields.rho[idx] = 1.0;

        // Inicialização do campo vetorial em regime permanente
        fields.ux[idx]  = U_INLET;
        fields.uy[idx]  = 0.0;
        fields.psi[idx] = Hx * (NX - x) + Hy * (NY - y);

        // Termos invariantes para o polinômio de Hermite
        double u_sq = U_INLET * U_INLET;
        double rho_local = 1.0;

        // Inicialização cinética rigorosa via estado de equilíbrio discreto
        for (int i = 0; i < 9; ++i) {
            // O produto escalar ci . u reduz-se a CX[i] * U_INLET pois uy = 0
            double cu = CX[i] * U_INLET;

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