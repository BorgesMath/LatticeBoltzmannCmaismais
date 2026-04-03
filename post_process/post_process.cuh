// post_process/post_process.cuh
#ifndef POST_PROCESS_CUH
#define POST_PROCESS_CUH

#include "../config/config.cuh"
#include <string>

// Cria o diretório timestampped e gera o log txt dos tensores
std::string init_post_processing();

// Orquestra a transferência VRAM -> RAM e a serialização VTK
void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields,
                double* h_phi, double* h_rho, double* h_ux, double* h_uy);

#endif // POST_PROCESS_CUH