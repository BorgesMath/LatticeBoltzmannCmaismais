#ifndef POST_PROCESS_CUH
#define POST_PROCESS_CUH

#include "../config/config.cuh"
#include <string>

// Inicializa o diretório de resultados e retorna o caminho
std::string init_post_processing();

// Assinatura atualizada: Exige apenas a densidade e as velocidades físicas
void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields,
                double* h_rho, double* h_ux, double* h_uy);

void save_metadata(const std::string& out_dir, double K, double U);
#endif // POST_PROCESS_CUH