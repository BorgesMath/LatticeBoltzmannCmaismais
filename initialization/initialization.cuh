#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../config/config.cuh"

// Assinatura atualizada (sem o argumento mode_m)
__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields);

#endif // INITIALIZATION_CUH