#ifndef POST_PROCESS_CUH
#define POST_PROCESS_CUH

#include "../config/config.cuh"
#include <string>

std::string init_post_processing();

void export_vtk(int step, const std::string& out_dir, Macro_Fields d_fields,
                double* h_phi, double* h_rho, double* h_ux, double* h_uy);

// Nova função de fechamento de log
void write_simulation_summary(const std::string& out_dir, double omega_theo,
                              double omega_num_mid, double omega_num_avg);

#endif // POST_PROCESS_CUH