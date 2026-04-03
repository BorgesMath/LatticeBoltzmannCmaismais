// multiphase/cahn_hilliard.cuh
#ifndef CAHN_HILLIARD_CUH
#define CAHN_HILLIARD_CUH

#include "../config/config.cuh"

// Orquestrador do sub-stepping que será evocado no loop principal
void solve_cahn_hilliard(Macro_Fields d_fields, dim3 numBlocks, dim3 threadsPerBlock);

#endif // CAHN_HILLIARD_CUH