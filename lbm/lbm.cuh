// lbm/lbm.cuh
#ifndef LBM_CUH
#define LBM_CUH

#include "../config/config.cuh"

__global__ void update_susceptibility_kernel(Macro_Fields fields, double chi_max, SimConfig cfg);
__global__ void lbm_collide_and_stream(LBM_Populations f_in, LBM_Populations f_out, Macro_Fields fields, SimConfig cfg);

// Novos kernels para o modelo de dupla população LBM-CH
__global__ void compute_chemical_potential_kernel(Macro_Fields fields, SimConfig cfg);
__global__ void lbm_collide_and_stream_phase(LBM_Populations_Phase g_in, LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg);
__global__ void update_macroscopic_phase_kernel(LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg);

#endif // LBM_CUH