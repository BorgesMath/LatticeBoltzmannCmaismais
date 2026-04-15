#ifndef CONFIG_CUH
#define CONFIG_CUH
#include <cuda_runtime.h>

constexpr bool IS_PERIODIC = false;
constexpr double BODY_FORCE_X = 0.0;

constexpr int NX = 200;
constexpr int NY = 50;
constexpr int NUM_NODES = NX * NY;
constexpr int SNAPSHOT_STEPS = 500;

constexpr double TAU_IN = 1.0;
constexpr double TAU_OUT = 1.0;
// double U_INLET = 0.001;
//constexpr double K_0 = 1.0;

constexpr double SIGMA = 0.0001;
constexpr double INTERFACE_WIDTH = 3.0;
constexpr double BETA = 3.0 * SIGMA * INTERFACE_WIDTH / 4.0;
constexpr double KAPPA = 3.0 * SIGMA * INTERFACE_WIDTH / 8.0;

static __constant__ double W_LBM[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static __constant__ int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static __constant__ int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static __constant__ int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

struct LBM_Populations {
    double *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8;
};

struct Macro_Fields {
    double *phi, *phi_new, *mu;
    double *psi, *rho, *ux, *uy, *chi_field, *K_field;
};

#endif // CONFIG_CUH
