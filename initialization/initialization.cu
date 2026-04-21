#include "initialization.cuh"
#include <cmath>

__global__ void init_fields_kernel(LBM_Populations f_in, LBM_Populations_Phase g_in, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = y * cfg.NX + x;

        // 1. Definição da Interface com Perturbação LSA
        double x_center = (double)cfg.NX * 0.40;
        double k_wave = 2.0 * PI * cfg.MODE_M / (double)cfg.NY;
        double dist = x_center + cfg.INITIAL_AMPLITUDE * cos(k_wave * y);

        // Perfil hiperbólico para o parâmetro de ordem phi
        double phi_val = -tanh((x - dist) / (cfg.INTERFACE_WIDTH / 2.0));
        fields.phi[idx] = phi_val;

        // 2. Inicialização de Campos Macroscópicos
        fields.K_field[idx] = cfg.K_0;
        fields.rho[idx] = 1.0;
        fields.ux[idx]  = cfg.U_INLET;
        fields.uy[idx]  = 0.0;

        // Campo magnético inicial (Potencial escalar psi)
        double angle_rad = cfg.H_ANGLE * PI / 180.0;
        double Hx = cfg.H0 * cos(angle_rad);
        double Hy = cfg.H0 * sin(angle_rad);
        fields.psi[idx] = Hx * (cfg.NX - x) + Hy * (cfg.NY - y);

        // 3. Distribuições de Equilíbrio (D2Q9)
        double u_sq = cfg.U_INLET * cfg.U_INLET;

        // No instante t=0, assumimos mu inicial nulo
        double mu_init = 0.0;

        // Assumindo tau_g = 1.0 na inicialização
        double tau_g = 1.0;
        double eta = cfg.M_MOBILITY / ((1.0 / 3.0) * (tau_g - 0.5));

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * cfg.U_INLET;

            // Equilíbrio para f_i (Navier-Stokes)
            double feq = W_LBM[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

            // Equilíbrio para g_i (Cahn-Hilliard - Chai & Zhao)
            double geq;
            if (i == 0) {
                geq = phi_val - (1.0 - W_LBM[0]) * eta * mu_init - W_LBM[0] * phi_val * (1.5 * u_sq);
            } else {
                geq = W_LBM[i] * eta * mu_init + W_LBM[i] * phi_val * (3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            }

            // Atribuição direta nos buffers de entrada
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