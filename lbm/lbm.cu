#include "lbm.cuh"
#include <cmath>

__device__ inline int get_idx(int x, int y) {
    return y * NX + x;
}

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = get_idx(x, y);
        fields.chi_field[idx] = chi_max * 0.5 * (1.0 + fields.phi[idx]);
    }
}

__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < NX && y < NY) {
        int idx = get_idx(x, y);

        double Fx = BODY_FORCE_X;
        double Fy = 0.0;
        double phi_c = fields.phi[idx];

        if (x > 0 && x < NX - 1 && y > 0 && y < NY - 1) {

            // =========================================================
            // A. FORÇA DE KORTEWEG (Réplica exata do lbm.py)
            // =========================================================
            double phi_R  = fields.phi[get_idx(x + 1, y)];
            double phi_L  = fields.phi[get_idx(x - 1, y)];
            double phi_T  = fields.phi[get_idx(x, y + 1)];
            double phi_B  = fields.phi[get_idx(x, y - 1)];

            // Derivada Central Simples (D2Q5) DEPOIS MELHORAR
            double dx_phi = 0.5 * (phi_R - phi_L);
            double dy_phi = 0.5 * (phi_T - phi_B);

            // Laplaciano Simples
            double lap_phi = phi_R + phi_L + phi_T + phi_B - 4.0 * phi_c;

            double mu_local = 4.0 * BETA * phi_c * (phi_c * phi_c - 1.0) - KAPPA * lap_phi;
            Fx += mu_local * dx_phi;
            Fy += mu_local * dy_phi;

            // =========================================================
            // B. FORÇA MAGNÉTICA DE KELVIN
            // =========================================================
            double psi_c = fields.psi[idx];
            double psi_R = fields.psi[get_idx(x + 1, y)];
            double psi_L = fields.psi[get_idx(x - 1, y)];
            double psi_T = fields.psi[get_idx(x, y + 1)];
            double psi_B = fields.psi[get_idx(x, y - 1)];
            double psi_TR = fields.psi[get_idx(x + 1, y + 1)];
            double psi_TL = fields.psi[get_idx(x - 1, y + 1)];
            double psi_BR = fields.psi[get_idx(x + 1, y - 1)];
            double psi_BL = fields.psi[get_idx(x - 1, y - 1)];

            double hx = -(1.0 / 6.0) * (2.0 * (psi_R - psi_L) + (psi_TR + psi_BR) - (psi_TL + psi_BL));
            double hy = -(1.0 / 6.0) * (2.0 * (psi_T - psi_B) + (psi_TR + psi_TL) - (psi_BR + psi_BL));

            double d2psi_dx2 = psi_R - 2.0 * psi_c + psi_L;
            double d2psi_dy2 = psi_T - 2.0 * psi_c + psi_B;
            double d2psi_dxy = 0.25 * (psi_TR - psi_TL - psi_BR + psi_BL);

            double chi = fields.chi_field[idx];
            Fx += chi * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy));
            Fy += chi * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2));
        }

        // =========================================================
        // C. MACROSCÓPICA E OPERADOR DE COLISÃO (GUO)
        // =========================================================
        double S_inv = fmax(0.0, fmin(1.0, (phi_c + 1.0) * 0.5));
        double tau = TAU_OUT + (TAU_IN - TAU_OUT) * S_inv;
        double omega = 1.0 / tau;
        double nu_local = (tau - 0.5) / 3.0;

        double k_local = fields.K_field[idx];
        double sigma_drag = nu_local / k_local;

        double f[9];
        f[0] = f_in.f0[idx]; f[1] = f_in.f1[idx]; f[2] = f_in.f2[idx];
        f[3] = f_in.f3[idx]; f[4] = f_in.f4[idx]; f[5] = f_in.f5[idx];
        f[6] = f_in.f6[idx]; f[7] = f_in.f7[idx]; f[8] = f_in.f8[idx];

        double rho_l = 0.0, ux_l = 0.0, uy_l = 0.0;
        for (int i = 0; i < 9; ++i) {
            rho_l += f[i];
            ux_l += f[i] * CX[i];
            uy_l += f[i] * CY[i];
        }

        double ux_star = (ux_l + 0.5 * Fx) / rho_l;
        double uy_star = (uy_l + 0.5 * Fy) / rho_l;

        double ux_phys = ux_star / (1.0 + 0.5 * sigma_drag);
        double uy_phys = uy_star / (1.0 + 0.5 * sigma_drag);
        double usq = ux_phys * ux_phys + uy_phys * uy_phys;

        // Recuperação exata da pressão mecânica do LBM
        fields.rho[idx] = rho_l;
        fields.ux[idx]  = ux_phys;
        fields.uy[idx]  = uy_phys;

        double Fx_total = Fx - (sigma_drag * rho_l * ux_phys);
        double Fy_total = Fy - (sigma_drag * rho_l * uy_phys);

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * ux_phys + CY[i] * uy_phys;
            double feq = W_LBM[i] * rho_l * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);

            double term1 = (CX[i] - ux_phys) * Fx_total + (CY[i] - uy_phys) * Fy_total;
            double term2 = cu * (CX[i] * Fx_total + CY[i] * Fy_total);
            double Si = W_LBM[i] * (1.0 - 0.5 * omega) * (3.0 * term1 + 9.0 * term2);

            double f_post = f[i] * (1.0 - omega) + omega * feq + Si;

            int be_x = x + CX[i];
            int be_y = y + CY[i];

            if (be_y >= 0 && be_y < NY) {
                if (IS_PERIODIC) {
                    int stream_x = (be_x + NX) % NX;
                    int stream_idx = get_idx(stream_x, be_y);
                    if(i==0) f_out.f0[stream_idx] = f_post; else if(i==1) f_out.f1[stream_idx] = f_post; else if(i==2) f_out.f2[stream_idx] = f_post; else if(i==3) f_out.f3[stream_idx] = f_post; else if(i==4) f_out.f4[stream_idx] = f_post; else if(i==5) f_out.f5[stream_idx] = f_post; else if(i==6) f_out.f6[stream_idx] = f_post; else if(i==7) f_out.f7[stream_idx] = f_post; else if(i==8) f_out.f8[stream_idx] = f_post;
                } else {
                    if (be_x >= 0 && be_x < NX) {
                        int stream_idx = get_idx(be_x, be_y);
                        if(i==0) f_out.f0[stream_idx] = f_post; else if(i==1) f_out.f1[stream_idx] = f_post; else if(i==2) f_out.f2[stream_idx] = f_post; else if(i==3) f_out.f3[stream_idx] = f_post; else if(i==4) f_out.f4[stream_idx] = f_post; else if(i==5) f_out.f5[stream_idx] = f_post; else if(i==6) f_out.f6[stream_idx] = f_post; else if(i==7) f_out.f7[stream_idx] = f_post; else if(i==8) f_out.f8[stream_idx] = f_post;
                    }
                }
            } else {
                int opp = OPP[i];
                if(opp==0) f_out.f0[idx] = f_post; else if(opp==1) f_out.f1[idx] = f_post; else if(opp==2) f_out.f2[idx] = f_post; else if(opp==3) f_out.f3[idx] = f_post; else if(opp==4) f_out.f4[idx] = f_post; else if(opp==5) f_out.f5[idx] = f_post; else if(opp==6) f_out.f6[idx] = f_post; else if(opp==7) f_out.f7[idx] = f_post; else if(opp==8) f_out.f8[idx] = f_post;
            }
        }
    }
}