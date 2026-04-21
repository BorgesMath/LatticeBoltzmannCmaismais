#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../config/config.cuh"

/**
 * @brief Kernel de inicialização para sistemas de dupla população.
 * @param f_in Populações hidrodinâmicas (D2Q9).
 * @param g_in Populações de campo de fase (D2Q9).
 * @param fields Estrutura de campos macroscópicos.
 * @param cfg Configurações da simulação.
 */
__global__ void init_fields_kernel(LBM_Populations f_in, LBM_Populations_Phase g_in, Macro_Fields fields, SimConfig cfg);

#endif // INITIALIZATION_CUH