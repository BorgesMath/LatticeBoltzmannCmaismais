#ifndef POISSON_CUH
#define POISSON_CUH

#include "../config/config.cuh"

void solve_poisson_magnetic(Macro_Fields fields, dim3 numBlocks2D, dim3 threadsPerBlock2D, SimConfig cfg);

#endif // POISSON_CUH