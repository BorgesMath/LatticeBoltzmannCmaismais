// Magnetismo/poisson.cuh
#ifndef POISSON_CUH
#define POISSON_CUH

#include "../config/config.cuh"

// Orquestrador do solver Red-Black SOR
void solve_poisson_magnetic(Macro_Fields fields, dim3 numBlocks2D, dim3 threadsPerBlock2D);

#endif // POISSON_CUH