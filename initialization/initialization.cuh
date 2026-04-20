#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../config/config.cuh"

__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields, SimConfig cfg);

#endif // INITIALIZATION_CUH