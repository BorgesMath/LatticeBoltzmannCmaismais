#ifndef ALLEN_CAHN_CUH
#define ALLEN_CAHN_CUH

#include "../config/config.cuh"

// Calcula o potencial químico para a Força de Korteweg (Navier-Stokes)
__global__ void compute_chemical_potential_kernel(Macro_Fields fields, SimConfig cfg);

// Evolução da fase com Fluxo Anti-difusivo de Allen-Cahn
__global__ void lbm_collide_and_stream_phase(LBM_Populations_Phase g_in, LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg);

// Atualização Macroscópica
__global__ void update_macroscopic_phase_kernel(LBM_Populations_Phase g_out, Macro_Fields fields, SimConfig cfg);

#endif // ALLEN_CAHN_CUH