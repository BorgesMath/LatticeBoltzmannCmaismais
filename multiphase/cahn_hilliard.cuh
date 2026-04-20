#ifndef CAHN_HILLIARD_CUH
#define CAHN_HILLIARD_CUH

#include "../config/config.cuh"

// Passagem de Macro_Fields obrigatória por referência (&) para preservação do swap D2D
void solve_cahn_hilliard(Macro_Fields& fields, dim3 numBlocks, dim3 threadsPerBlock, SimConfig cfg);

#endif // CAHN_HILLIARD_CUH