#include "lbm.cuh"
#include <cmath>

__device__ inline int get_idx(int x, int y, int NX_dim) { return y * NX_dim + x; }
__device__ inline int x_R(int x, int NX) { return min(x + 1, NX - 1); }
__device__ inline int x_L(int x, int NX) { return max(x - 1, 0); }
__device__ inline int y_T(int y, int NY, bool per) { return per ? (y + 1) % NY : min(y + 1, NY - 1); }
__device__ inline int y_B(int y, int NY, bool per) { return per ? (y - 1 + NY) % NY : max(y - 1, 0); }
__device__ inline int y_T_clamp(int y, int NY) { return min(y + 1, NY - 1); }
__device__ inline int y_B_clamp(int y, int NY) { return max(y - 1, 0); }

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max, SimConfig cfg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cfg.NX && y < cfg.NY) fields.chi_field[get_idx(x, y, cfg.NX)] = chi_max * 0.5 * (1.0 + fields.phi[get_idx(x, y, cfg.NX)]);
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

            int R  = get_idx(x_R(x, cfg.NX), y, cfg.NX);
            int L  = get_idx(x_L(x, cfg.NX), y, cfg.NX);
            int T  = get_idx(x, y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int B  = get_idx(x, y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int TR = get_idx(x_R(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int TL = get_idx(x_L(x, cfg.NX), y_T(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int BR = get_idx(x_R(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);
            int BL = get_idx(x_L(x, cfg.NX), y_B(y, cfg.NY, cfg.IS_PERIODIC), cfg.NX);

            double dx_phi = (1.0 / 6.0) * (2.0 * (fields.phi[R] - fields.phi[L]) + (fields.phi[TR] + fields.phi[BR]) - (fields.phi[TL] + fields.phi[BL]));
            double dy_phi = (1.0 / 6.0) * (2.0 * (fields.phi[T] - fields.phi[B]) + (fields.phi[TR] + fields.phi[TL]) - (fields.phi[BR] + fields.phi[BL]));

            // O Potencial mu foi previamente calculado no allen_cahn.cu
            double mu_c = fields.mu[idx];
            Fx += mu_c * dx_phi;
            Fy += mu_c * dy_phi;

            int mT  = get_idx(x, y_T_clamp(y, cfg.NY), cfg.NX);
            int mB  = get_idx(x, y_B_clamp(y, cfg.NY), cfg.NX);
            int mTR = get_idx(x_R(x, cfg.NX), y_T_clamp(y, cfg.NY), cfg.NX);
            int mTL = get_idx(x_L(x, cfg.NX), y_T_clamp(y, cfg.NY), cfg.NX);
            int mBR = get_idx(x_R(x, cfg.NX), y_B_clamp(y, cfg.NY), cfg.NX);
            int mBL = get_idx(x_L(x, cfg.NX), y_B_clamp(y, cfg.NY), cfg.NX);

            double psi_c = fields.psi[idx];
            double psi_R = fields.psi[R];  double psi_L = fields.psi[L];
            double psi_T = fields.psi[mT]; double psi_B = fields.psi[mB];

            double hx = -(1.0 / 6.0) * (2.0 * (psi_R - psi_L) + (fields.psi[mTR] + fields.psi[mBR]) - (fields.psi[mTL] + fields.psi[mBL]));
            double hy = -(1.0 / 6.0) * (2.0 * (psi_T - psi_B) + (fields.psi[mTR] + fields.psi[mTL]) - (fields.psi[mBR] + fields.psi[mBL]));

            double d2psi_dx2 = psi_R - 2.0 * psi_c + psi_L;
            double d2psi_dy2 = psi_T - 2.0 * psi_c + psi_B;
            double d2psi_dxy = 0.25 * (fields.psi[mTR] - fields.psi[mTL] - fields.psi[mBR] + fields.psi[mBL]);

            double chi = fields.chi_field[idx];
            Fx += chi * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy));
            Fy += chi * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2));
        }

        double S_inv = fmax(0.0, fmin(1.0, (phi_c + 1.0) * 0.5));
        double tau = cfg.TAU_OUT + (cfg.TAU_IN - cfg.TAU_OUT) * S_inv;
        double omega = 1.0 / tau;
        double nu_local = (tau - 0.5) / 3.0;

        double sigma_drag = nu_local / fields.K_field[idx];

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

        double ux_phys = ((ux_l + 0.5 * Fx) / rho_l) / (1.0 + 0.5 * sigma_drag);
        double uy_phys = ((uy_l + 0.5 * Fy) / rho_l) / (1.0 + 0.5 * sigma_drag);
        double usq = ux_phys * ux_phys + uy_phys * uy_phys;

        fields.rho[idx] = rho_l; fields.ux[idx] = ux_phys; fields.uy[idx] = uy_phys;

        double Fx_total = Fx - (sigma_drag * rho_l * ux_phys);
        double Fy_total = Fy - (sigma_drag * rho_l * uy_phys);

        for (int i = 0; i < 9; ++i) {
            double cu = CX[i] * ux_phys + CY[i] * uy_phys;
            double feq = W_LBM[i] * rho_l * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);

            double Si = W_LBM[i] * (1.0 - 0.5 * omega) * (3.0 * ((CX[i] - ux_phys) * Fx_total + (CY[i] - uy_phys) * Fy_total) + 9.0 * cu * (CX[i] * Fx_total + CY[i] * Fy_total));

            double f_post = f[i] * (1.0 - omega) + omega * feq + Si;

            int be_x = x + CX[i];
            int be_y = y + CY[i];

            if (cfg.IS_PERIODIC) be_y = (be_y % cfg.NY + cfg.NY) % cfg.NY;

            if (be_x >= 0 && be_x < cfg.NX) {
                if (!cfg.IS_PERIODIC && (be_y < 0 || be_y >= cfg.NY)) {
                    int opp = OPP[i];
                    if(opp==0) f_out.f0[idx] = f_post; else if(opp==1) f_out.f1[idx] = f_post;
                    else if(opp==2) f_out.f2[idx] = f_post; else if(opp==3) f_out.f3[idx] = f_post;
                    else if(opp==4) f_out.f4[idx] = f_post; else if(opp==5) f_out.f5[idx] = f_post;
                    else if(opp==6) f_out.f6[idx] = f_post; else if(opp==7) f_out.f7[idx] = f_post;
                    else if(opp==8) f_out.f8[idx] = f_post;
                } else {
                    int stream_idx = get_idx(be_x, be_y, cfg.NX);
                    if(i==0) f_out.f0[stream_idx] = f_post; else if(i==1) f_out.f1[stream_idx] = f_post;
                    else if(i==2) f_out.f2[stream_idx] = f_post; else if(i==3) f_out.f3[stream_idx] = f_post;
                    else if(i==4) f_out.f4[stream_idx] = f_post; else if(i==5) f_out.f5[stream_idx] = f_post;
                    else if(i==6) f_out.f6[stream_idx] = f_post; else if(i==7) f_out.f7[stream_idx] = f_post;
                    else if(i==8) f_out.f8[stream_idx] = f_post;
                }
            }
        }
    }
}