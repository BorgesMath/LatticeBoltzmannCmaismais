#include "lbm.cuh"
#include <cmath>

__device__ inline int get_idx(int x, int y, int NX_dim) {
    return y * NX_dim + x;
}

// =====================================================================
// KERNELS ORIGINAIS: MAGNETISMO E HIDRODINÂMICA (f_i)
// =====================================================================

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);
        fields.chi_field[idx] = chi_max * 0.5 * (1.0 + fields.phi[idx]);
    }
}

__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);

        double Fx = cfg.BODY_FORCE_X;
        double Fy = 0.0;
        double phi_c = fields.phi[idx];

        if (x > 0 && x < cfg.NX - 1 && y > 0 && y < cfg.NY - 1) {
            double phi_R  = fields.phi[get_idx(x + 1, y, cfg.NX)];
            double phi_L  = fields.phi[get_idx(x - 1, y, cfg.NX)];
            double phi_T  = fields.phi[get_idx(x, y + 1, cfg.NX)];
            double phi_B  = fields.phi[get_idx(x, y - 1, cfg.NX)];
            double phi_TR = fields.phi[get_idx(x + 1, y + 1, cfg.NX)];
            double phi_TL = fields.phi[get_idx(x - 1, y + 1, cfg.NX)];
            double phi_BR = fields.phi[get_idx(x + 1, y - 1, cfg.NX)];
            double phi_BL = fields.phi[get_idx(x - 1, y - 1, cfg.NX)];

            double dx_phi = (1.0 / 6.0) * (2.0 * (phi_R - phi_L) + (phi_TR + phi_BR) - (phi_TL + phi_BL));
            double dy_phi = (1.0 / 6.0) * (2.0 * (phi_T - phi_B) + (phi_TR + phi_TL) - (phi_BR + phi_BL));
            double lap_phi = (1.0 / 6.0) * (4.0 * (phi_R + phi_L + phi_T + phi_B) +
                                            1.0 * (phi_TR + phi_TL + phi_BR + phi_BL) - 20.0 * phi_c);

            double mu_c = 4.0 * cfg.BETA * phi_c * (phi_c * phi_c - 1.0) - cfg.KAPPA * lap_phi;
            Fx += mu_c * dx_phi;
            Fy += mu_c * dy_phi;

            double psi_c = fields.psi[idx];
            double psi_R = fields.psi[get_idx(x + 1, y, cfg.NX)];
            double psi_L = fields.psi[get_idx(x - 1, y, cfg.NX)];
            double psi_T = fields.psi[get_idx(x, y + 1, cfg.NX)];
            double psi_B = fields.psi[get_idx(x, y - 1, cfg.NX)];
            double psi_TR = fields.psi[get_idx(x + 1, y + 1, cfg.NX)];
            double psi_TL = fields.psi[get_idx(x - 1, y + 1, cfg.NX)];
            double psi_BR = fields.psi[get_idx(x + 1, y - 1, cfg.NX)];
            double psi_BL = fields.psi[get_idx(x - 1, y - 1, cfg.NX)];

            double hx = -(1.0 / 6.0) * (2.0 * (psi_R - psi_L) + (psi_TR + psi_BR) - (psi_TL + psi_BL));
            double hy = -(1.0 / 6.0) * (2.0 * (psi_T - psi_B) + (psi_TR + psi_TL) - (psi_BR + psi_BL));

            double d2psi_dx2 = psi_R - 2.0 * psi_c + psi_L;
            double d2psi_dy2 = psi_T - 2.0 * psi_c + psi_B;
            double d2psi_dxy = 0.25 * (psi_TR - psi_TL - psi_BR + psi_BL);

            double chi = fields.chi_field[idx];
            Fx += chi * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy));
            Fy += chi * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2));
        }

        double S_inv = fmax(0.0, fmin(1.0, (phi_c + 1.0) * 0.5));
        double tau = cfg.TAU_OUT + (cfg.TAU_IN - cfg.TAU_OUT) * S_inv;
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

            if (be_y >= 0 && be_y < cfg.NY) {
                if (cfg.IS_PERIODIC) {
                    int stream_x = (be_x + cfg.NX) % cfg.NX;
                    int stream_idx = get_idx(stream_x, be_y, cfg.NX);

                    if(i==0) f_out.f0[stream_idx] = f_post;
                    else if(i==1) f_out.f1[stream_idx] = f_post;
                    else if(i==2) f_out.f2[stream_idx] = f_post;
                    else if(i==3) f_out.f3[stream_idx] = f_post;
                    else if(i==4) f_out.f4[stream_idx] = f_post;
                    else if(i==5) f_out.f5[stream_idx] = f_post;
                    else if(i==6) f_out.f6[stream_idx] = f_post;
                    else if(i==7) f_out.f7[stream_idx] = f_post;
                    else if(i==8) f_out.f8[stream_idx] = f_post;
                } else {
                    if (be_x >= 0 && be_x < cfg.NX) {
                        int stream_idx = get_idx(be_x, be_y, cfg.NX);
                        if(i==0) f_out.f0[stream_idx] = f_post;
                        else if(i==1) f_out.f1[stream_idx] = f_post;
                        else if(i==2) f_out.f2[stream_idx] = f_post;
                        else if(i==3) f_out.f3[stream_idx] = f_post;
                        else if(i==4) f_out.f4[stream_idx] = f_post;
                        else if(i==5) f_out.f5[stream_idx] = f_post;
                        else if(i==6) f_out.f6[stream_idx] = f_post;
                        else if(i==7) f_out.f7[stream_idx] = f_post;
                        else if(i==8) f_out.f8[stream_idx] = f_post;
                    }
                }
            } else {
                int opp = OPP[i];
                if(opp==0) f_out.f0[idx] = f_post;
                else if(opp==1) f_out.f1[idx] = f_post;
                else if(opp==2) f_out.f2[idx] = f_post;
                else if(opp==3) f_out.f3[idx] = f_post;
                else if(opp==4) f_out.f4[idx] = f_post;
                else if(opp==5) f_out.f5[idx] = f_post;
                else if(opp==6) f_out.f6[idx] = f_post;
                else if(opp==7) f_out.f7[idx] = f_post;
                else if(opp==8) f_out.f8[idx] = f_post;
            }
        }
    }
}

// =====================================================================
// NOVOS KERNELS: CAMPO DE FASE CAHN-HILLIARD (g_i)
// =====================================================================

__global__ void compute_chemical_potential_kernel(Macro_Fields fields, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);

        int xR = min(x + 1, cfg.NX - 1);
        int xL = max(x - 1, 0);
        int yT = min(y + 1, cfg.NY - 1);
        int yB = max(y - 1, 0);

        double phi_c  = fields.phi[idx];
        double phi_R  = fields.phi[get_idx(xR, y, cfg.NX)];
        double phi_L  = fields.phi[get_idx(xL, y, cfg.NX)];
        double phi_T  = fields.phi[get_idx(x, yT, cfg.NX)];
        double phi_B  = fields.phi[get_idx(x, yB, cfg.NX)];
        double phi_TR = fields.phi[get_idx(xR, yT, cfg.NX)];
        double phi_TL = fields.phi[get_idx(xL, yT, cfg.NX)];
        double phi_BR = fields.phi[get_idx(xR, yB, cfg.NX)];
        double phi_BL = fields.phi[get_idx(xL, yB, cfg.NX)];

        double lap_phi = (1.0 / 6.0) * (4.0 * (phi_R + phi_L + phi_T + phi_B) +
                                        1.0 * (phi_TR + phi_TL + phi_BR + phi_BL) - 20.0 * phi_c);

        fields.mu[idx] = 4.0 * cfg.BETA * phi_c * (phi_c * phi_c - 1.0) - cfg.KAPPA * lap_phi;
    }
}

__global__ void lbm_collide_and_stream_phase(LBM_Populations_Phase g_in, LBM_Populations_Phase g_out, Macro_Fields fields, double tau_g, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cfg.NX && y < cfg.NY) {
        int idx = get_idx(x, y, cfg.NX);

        double phi_c = fields.phi[idx];
        double mu_c  = fields.mu[idx]; // Leitura direta do potencial químico local
        double ux = fields.ux[idx];
        double uy = fields.uy[idx];
        double usq = ux * ux + uy * uy;

        double omega_g = 1.0 / tau_g;

        // Fator de ajuste de mobilidade (Derivado da expansão Chapman-Enskog LBM-CH)
        // M = eta * c_s^2 * (tau_g - 0.5) * dt  =>  eta = M / (1/3 * (tau_g - 0.5))
        double eta = cfg.M_MOBILITY / ((1.0 / 3.0) * (tau_g - 0.5));

        double g[9];
        g[0] = g_in.g0[idx]; g[1] = g_in.g1[idx]; g[2] = g_in.g2[idx];
        g[3] = g_in.g3[idx]; g[4] = g_in.g4[idx]; g[5] = g_in.g5[idx];
        g[6] = g_in.g6[idx]; g[7] = g_in.g7[idx]; g[8] = g_in.g8[idx];

        for (int i = 0; i < 9; ++i) {
            double geq;

            // Formulação Chai & Zhao (2013) para Cahn-Hilliard
            if (i == 0) {
                // População de repouso cancela a injeção de massa de mu para manter integral(phi) constante
                geq = phi_c - (1.0 - W_LBM[0]) * eta * mu_c - W_LBM[0] * phi_c * (1.5 * usq);
            } else {
                double cu = CX[i] * ux + CY[i] * uy;
                geq = W_LBM[i] * eta * mu_c + W_LBM[i] * phi_c * (3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
            }

            // Colisão BGK pura (Sem termo fonte S_i)
            double g_post = g[i] * (1.0 - omega_g) + omega_g * geq;

            // Streaming Padrão
            int be_x = x + CX[i];
            int be_y = y + CY[i];

            if (be_y >= 0 && be_y < cfg.NY) {
                if (cfg.IS_PERIODIC) {
                    int stream_x = (be_x + cfg.NX) % cfg.NX;
                    int stream_idx = get_idx(stream_x, be_y, cfg.NX);

                    if(i==0) g_out.g0[stream_idx] = g_post;
                    else if(i==1) g_out.g1[stream_idx] = g_post;
                    else if(i==2) g_out.g2[stream_idx] = g_post;
                    else if(i==3) g_out.g3[stream_idx] = g_post;
                    else if(i==4) g_out.g4[stream_idx] = g_post;
                    else if(i==5) g_out.g5[stream_idx] = g_post;
                    else if(i==6) g_out.g6[stream_idx] = g_post;
                    else if(i==7) g_out.g7[stream_idx] = g_post;
                    else if(i==8) g_out.g8[stream_idx] = g_post;
                } else {
                    if (be_x >= 0 && be_x < cfg.NX) {
                        int stream_idx = get_idx(be_x, be_y, cfg.NX);
                        if(i==0) g_out.g0[stream_idx] = g_post;
                        else if(i==1) g_out.g1[stream_idx] = g_post;
                        else if(i==2) g_out.g2[stream_idx] = g_post;
                        else if(i==3) g_out.g3[stream_idx] = g_post;
                        else if(i==4) g_out.g4[stream_idx] = g_post;
                        else if(i==5) g_out.g5[stream_idx] = g_post;
                        else if(i==6) g_out.g6[stream_idx] = g_post;
                        else if(i==7) g_out.g7[stream_idx] = g_post;
                        else if(i==8) g_out.g8[stream_idx] = g_post;
                    }
                }
            } else {
                int opp = OPP[i];
                if(opp==0) g_out.g0[idx] = g_post;
                else if(opp==1) g_out.g1[idx] = g_post;
                else if(opp==2) g_out.g2[idx] = g_post;
                else if(opp==3) g_out.g3[idx] = g_post;
                else if(opp==4) g_out.g4[idx] = g_post;
                else if(opp==5) g_out.g5[idx] = g_post;
                else if(opp==6) g_out.g6[idx] = g_post;
                else if(opp==7) g_out.g7[idx] = g_post;
                else if(opp==8) g_out.g8[idx] = g_post;
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