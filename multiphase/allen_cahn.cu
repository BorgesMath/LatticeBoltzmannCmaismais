#include "allen_cahn.cuh"
#include <cmath>

__device__ inline int get_idx(int x, int y, int NX_dim) { return y * NX_dim + x; }

__device__ inline int x_R(int x, int NX) { return min(x + 1, NX - 1); }
__device__ inline int x_L(int x, int NX) { return max(x - 1, 0); }
__device__ inline int y_T(int y, int NY, bool per) { return per ? (y + 1) % NY : min(y + 1, NY - 1); }
__device__ inline int y_B(int y, int NY, bool per) { return per ? (y - 1 + NY) % NY : max(y - 1, 0); }

__global__ void compute_chemical_potential_kernel(Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);

        int R  = get_idx(x_R(x, cfg.NX), y, cfg.NX);
        int L  = get_idx(x_L(x, cfg.NX), y, cfg.NX);
        int T  = get_idx(x, y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
        int B  = get_idx(x, y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
        int TR = get_idx(x_R(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
        int TL = get_idx(x_L(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
        int BR = get_idx(x_R(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
        int BL = get_idx(x_L(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);

        double phi_c  = fields.phi[idx];
        double lap_phi = (1.0 / 6.0) * (4.0 * (fields.phi[R] + fields.phi[L] + fields.phi[T] + fields.phi[B]) +
                                        1.0 * (fields.phi[TR] + fields.phi[TL] + fields.phi[BR] + fields.phi[BL]) - 20.0 * phi_c);

        // O potencial é mantido exclusivamente para calcular a força capilar no LBM hidrodinâmico
        fields.mu[idx] = 4.0 * cfg.BETA * phi_c * (phi_c * phi_c - 1.0) - cfg.KAPPA * lap_phi;
    }
}

__global__ void lbm_collide_and_stream_phase(LBM_Populations_Phase g_in, LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);

        double phi_c = fields.phi[idx];
        double ux = fields.ux[idx];
        double uy = fields.uy[idx];
        double usq = ux * ux + uy * uy;

        double dx_phi = 0.0, dy_phi = 0.0;

        if (x > 0 && x < cfg.NX - 1) {
            int R  = get_idx(x_R(x, cfg.NX), y, cfg.NX);
            int L  = get_idx(x_L(x, cfg.NX), y, cfg.NX);
            int T  = get_idx(x, y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int B  = get_idx(x, y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int TR = get_idx(x_R(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int TL = get_idx(x_L(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int BR = get_idx(x_R(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int BL = get_idx(x_L(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);

            dx_phi = (1.0 / 3.0) * (fields.phi[R] - fields.phi[L]) + (1.0 / 12.0) * (fields.phi[TR] - fields.phi[TL] + fields.phi[BR] - fields.phi[BL]);
            dy_phi = (1.0 / 3.0) * (fields.phi[T] - fields.phi[B]) + (1.0 / 12.0) * (fields.phi[TR] + fields.phi[TL] - fields.phi[BR] - fields.phi[BL]);
        }

        // Operador Conservativo de Allen-Cahn (Zheng et al.)
        double grad_mag = sqrt(dx_phi * dx_phi + dy_phi * dy_phi) + 1e-12;
        double nx = dx_phi / grad_mag;
        double ny = dy_phi / grad_mag;

        double M = cfg.M_MOBILITY;
        double D_lbm = 1.0 / 6.0; // Fixo para tau_g = 1.0
        double W = cfg.INTERFACE_WIDTH;

        // Força compressiva termodinâmica
        double compress_factor = M * (2.0 * fmax(1.0 - phi_c * phi_c, 0.0)) / W;

        // Fluxo Q injetado anula a difusão da malha e aplica a mobilidade e compressão de Allen-Cahn
        double Qx = (M - D_lbm) * dx_phi - compress_factor * nx;
        double Qy = (M - D_lbm) * dy_phi - compress_factor * ny;

        double g[9];
        g[0] = g_in.g0[idx]; g[1] = g_in.g1[idx]; g[2] = g_in.g2[idx];
        g[3] = g_in.g3[idx]; g[4] = g_in.g4[idx]; g[5] = g_in.g5[idx];
        g[6] = g_in.g6[idx]; g[7] = g_in.g7[idx]; g[8] = g_in.g8[idx];

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * ux + CY[i] * uy;

            // Equilíbrio Termodinâmico Estrito (Sem gambiarras acústicas)
            double geq = W_LBM[i] * phi_c * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);

            // Injeção de Fluxo Macroscópico (1.5 = 0.5 * 3.0 para tau=1.0)
            double Si = 1.5 * W_LBM[i] * (CX[i] * Qx + CY[i] * Qy);

            double g_post = g[i] * 0.0 + 1.0 * geq + Si; // tau_g = 1.0 -> omega = 1.0

            int be_x = x + CX[i];
            int be_y = y + CY[i];

            if (cfg.IS_PERIODIC) be_y = (be_y % cfg.NY + cfg.NY) % cfg.NY;

            if (be_x >= 0 && be_x < cfg.NX) {
                if (!cfg.IS_PERIODIC && (be_y < 0 || be_y >= cfg.NY)) {
                    int opp = OPP[i];
                    if(opp==0) g_out.g0[idx] = g_post; else if(opp==1) g_out.g1[idx] = g_post;
                    else if(opp==2) g_out.g2[idx] = g_post; else if(opp==3) g_out.g3[idx] = g_post;
                    else if(opp==4) g_out.g4[idx] = g_post; else if(opp==5) g_out.g5[idx] = g_post;
                    else if(opp==6) g_out.g6[idx] = g_post; else if(opp==7) g_out.g7[idx] = g_post;
                    else if(opp==8) g_out.g8[idx] = g_post;
                } else {
                    int stream_idx = get_idx(be_x, be_y, cfg.NX);
                    if(i==0) g_out.g0[stream_idx] = g_post; else if(i==1) g_out.g1[stream_idx] = g_post;
                    else if(i==2) g_out.g2[stream_idx] = g_post; else if(i==3) g_out.g3[stream_idx] = g_post;
                    else if(i==4) g_out.g4[stream_idx] = g_post; else if(i==5) g_out.g5[stream_idx] = g_post;
                    else if(i==6) g_out.g6[stream_idx] = g_post; else if(i==7) g_out.g7[stream_idx] = g_post;
                    else if(i==8) g_out.g8[stream_idx] = g_post;
                }
            }
        }
    }
}

__global__ void update_macroscopic_phase_kernel(LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);
        double phi_new = g_out.g0[idx] + g_out.g1[idx] + g_out.g2[idx] +
                         g_out.g3[idx] + g_out.g4[idx] + g_out.g5[idx] +
                         g_out.g6[idx] + g_out.g7[idx] + g_out.g8[idx];
        fields.phi[idx] = phi_new;
    }
}