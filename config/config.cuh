#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include <string>

constexpr double PI = 3.14159265358979323846;

static __constant__ double W_LBM[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static __constant__ int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static __constant__ int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static __constant__ int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

struct SimConfig {
    char case_name[64]; // Subsitui std::string para compatibilidade estrita POD (Plain Old Data)
    int NX;
    int NY;
    int NUM_NODES;
    int SNAPSHOT_STEPS;

    double TAU_IN;
    double TAU_OUT;
    double U_INLET;
    double K_0;

    double M_MOBILITY;
    int CH_SUBSTEPS;
    double DT_CH;
    double SIGMA;
    double INTERFACE_WIDTH;
    double BETA;
    double KAPPA;

    double H0;
    double H_ANGLE;
    double SOR_OMEGA;
    int SOR_ITERATIONS;

    double INITIAL_AMPLITUDE;
    int MODE_M;
    double BODY_FORCE_X;
    bool IS_PERIODIC;
};

struct LBM_Populations {
    double *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8;
};

struct Macro_Fields {
    double *phi, *phi_new, *mu, *psi, *rho, *ux, *uy, *chi_field, *K_field;
};

#endif // CONFIG_CUH