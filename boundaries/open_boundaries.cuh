#ifndef OPEN_BOUNDARIES_CUH
#define OPEN_BOUNDARIES_CUH

#include "../config/config.cuh"

void apply_open_boundaries(LBM_Populations d_f_out, LBM_Populations_Phase d_g_out, Macro_Fields d_fields, int NY_dim, SimConfig cfg);

#endif // OPEN_BOUNDARIES_CUH