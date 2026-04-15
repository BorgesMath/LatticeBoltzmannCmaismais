#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../config/config.cuh"

// Assinatura exige o parâmetro K_0_val
__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields, double K_0_val);

#endif // INITIALIZATION_CUH