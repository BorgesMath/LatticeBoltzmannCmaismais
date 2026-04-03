// boundaries/open_boundaries.cuh
#ifndef OPEN_BOUNDARIES_CUH
#define OPEN_BOUNDARIES_CUH

#include "../config/config.cuh"

// Orquestrador do Kernel de contorno 1D
void apply_open_boundaries(LBM_Populations d_f_out, Macro_Fields d_fields, int NY_dim);

#endif // OPEN_BOUNDARIES_CUH