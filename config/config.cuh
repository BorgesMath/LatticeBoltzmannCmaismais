#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>

// =========================================================
// CONTROLE DE TOPOLOGIA E FORÇAMENTO UNIFICADO
// =========================================================
constexpr bool IS_PERIODIC = true; // true: Validação (Darcy), false: Produção (Saffman-Taylor)


// =========================================================
// 0. CONSTANTES MATEMÁTICAS
// =========================================================
constexpr double PI = 3.14159265358979323846;

// =========================================================
// 1. PARÂMETROS DE SIMULAÇÃO E TOPOLOGIA
// =========================================================
constexpr int NX = 400;
constexpr int NY = 400;
constexpr int NUM_NODES = NX * NY;
constexpr int SNAPSHOT_STEPS = 500;

// =========================================================
// 2. HIDRODINÂMICA E CINEMÁTICA
// =========================================================
constexpr double TAU_IN = 1.0;
constexpr double TAU_OUT = 1.0;
constexpr double U_INLET = 0.0;
constexpr double K_0 = 1.0e15;
constexpr double BODY_FORCE_X = 0; // Fx constante. Deixe 0.0 para Saffman-Taylor.



// =========================================================
// 3. TERMODINÂMICA DE INTERFACE (CAHN-HILLIARD)
// =========================================================
constexpr double M_MOBILITY = 0.002;
constexpr int CH_SUBSTEPS = 10;
constexpr double DT_CH = 1.0 /  (double)CH_SUBSTEPS;

constexpr double SIGMA = 0.0001;
constexpr double INTERFACE_WIDTH = 5.0;
//constexpr double BETA = 3.0 * (3.0 * SIGMA) / (4.0 * INTERFACE_WIDTH);
//constexpr double KAPPA = 3.0 * SIGMA * INTERFACE_WIDTH / 8.0;


constexpr double BETA = 0.000009375; // 9.375e-6
constexpr double KAPPA = 0.0003;     // 3.0e-4

// =========================================================
// 4. MAGNETOSTÁTICA E CONTROLE DE SOLVER
// =========================================================
constexpr double H0 = 0.0;
constexpr double H_ANGLE = 0.0;
constexpr double SOR_OMEGA = 1.85;
constexpr int SOR_ITERATIONS = 15;

// =========================================================
// 5. CONDIÇÕES INICIAIS DA PERTURBAÇÃO
// =========================================================
constexpr double INITIAL_AMPLITUDE = 2.0;
constexpr int MODE_M = 4; // Modo de perturbação (Define o número de onda k)

// =========================================================
// 6. TENSORES DO MODELO LBM D2Q9 (Memória Constante)
// =========================================================
// O modificador 'static' previne erros de múltipla definição no Linker (LNK2005)
// ao incluir este cabeçalho em vários arquivos .cu simultaneamente.
static __constant__ double W_LBM[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static __constant__ int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static __constant__ int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static __constant__ int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


// =========================================================
// 7. ESTRUTURAS DE DADOS GLOBAIS (SoA)
// =========================================================

struct LBM_Populations {
    double *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8;
};

struct Macro_Fields {
    // Campos do Cahn-Hilliard rigorosamente declarados aqui:
    double *phi, *phi_new, *mu;

    // Campos Macro e Magnetostática:
    double *psi, *rho, *ux, *uy, *chi_field, *K_field;
};

#endif // CONFIG_CUH