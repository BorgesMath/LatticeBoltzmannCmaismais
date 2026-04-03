// initialization/initialization.cuh
#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../config/config.cuh"

// Declaração do Kernel de Inicialização
__global__ void init_fields_kernel(LBM_Populations f_in, Macro_Fields fields, int mode_m);

#endif // INITIALIZATION_CUH