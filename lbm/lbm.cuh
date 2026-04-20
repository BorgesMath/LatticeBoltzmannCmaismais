#ifndef LBM_CUH
#define LBM_CUH

#include "../config/config.cuh"

// Atualiza a susceptibilidade magnética com base no parâmetro de ordem
__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max);

// Operador central LBM: Colisão BGK, Forças Macroscópicas e Push Streaming
__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields);

#endif