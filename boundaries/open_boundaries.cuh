#ifndef OPEN_BOUNDARIES_CUH
#define OPEN_BOUNDARIES_CUH

#include "../config/config.cuh"

__global__ void open_boundaries_kernel(LBM_Populations f_out, Macro_Fields fields, double U_in_val);

void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim, double U_in_val);

#endif // OPEN_BOUNDARIES_CUH