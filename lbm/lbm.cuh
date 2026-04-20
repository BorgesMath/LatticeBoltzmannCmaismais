#ifndef LBM_CUH
#define LBM_CUH

#include "../config/config.cuh"

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max, SimConfig cfg);
__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields, SimConfig cfg);

#endif // LBM_CUH