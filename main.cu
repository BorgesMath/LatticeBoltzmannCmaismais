#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"
#include "lbm/lbm.cuh"

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "Erro CUDA: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

void allocate_populations(LBM_Populations* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->f0), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f2), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f4), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f6), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f8), bytes));
}

void allocate_macro_fields(Macro_Fields* f, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(f->phi), bytes)); CUDA_CHECK(cudaMalloc((void**)&(f->phi_new), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->mu), bytes));  CUDA_CHECK(cudaMalloc((void**)&(f->psi), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->rho), bytes)); CUDA_CHECK(cudaMalloc((void**)&(f->ux), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->uy), bytes));  CUDA_CHECK(cudaMalloc((void**)&(f->chi_field), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->K_field), bytes));
}

void free_populations(LBM_Populations* p) {
    CUDA_CHECK(cudaFree(p->f0)); CUDA_CHECK(cudaFree(p->f1)); CUDA_CHECK(cudaFree(p->f2));
    CUDA_CHECK(cudaFree(p->f3)); CUDA_CHECK(cudaFree(p->f4)); CUDA_CHECK(cudaFree(p->f5));
    CUDA_CHECK(cudaFree(p->f6)); CUDA_CHECK(cudaFree(p->f7)); CUDA_CHECK(cudaFree(p->f8));
}

void free_macro_fields(Macro_Fields* f) {
    CUDA_CHECK(cudaFree(f->phi)); CUDA_CHECK(cudaFree(f->phi_new)); CUDA_CHECK(cudaFree(f->mu));
    CUDA_CHECK(cudaFree(f->psi)); CUDA_CHECK(cudaFree(f->rho)); CUDA_CHECK(cudaFree(f->ux));
    CUDA_CHECK(cudaFree(f->uy));  CUDA_CHECK(cudaFree(f->chi_field)); CUDA_CHECK(cudaFree(f->K_field));
}

void swap_populations(LBM_Populations* p1, LBM_Populations* p2) {
    LBM_Populations temp = *p1; *p1 = *p2; *p2 = temp;
}

void print_progress_bar(int step, int total) {
    float progress = (float)step / total;
    int barWidth = 50;
    std::cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% | Iteracao: " << step << "  ";
    std::cout.flush();
    if (step == total) std::cout << std::endl;
}

int main() {
    LBM_Populations d_f_in, d_f_out;
    Macro_Fields d_fields;
    size_t mem_size = NUM_NODES * sizeof(double);

    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    std::string output_dir = init_post_processing();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields);
    CUDA_CHECK(cudaDeviceSynchronize());

    int max_iter = 50000;

    std::cout << "Iniciando Simulacao de Poiseuille..." << std::endl;

    for (int t = 0; t <= max_iter; ++t) {

        // Operador LBM Puro
        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Imposição de Fronteiras Abertas (Inlet e Outlet)
        apply_open_boundaries(d_f_out, d_fields, NY);
        CUDA_CHECK(cudaDeviceSynchronize());

        swap_populations(&d_f_in, &d_f_out);

        if (t % SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_rho, h_ux, h_uy);
        }

        if (t % 100 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter);
        }
    }

    free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out); free_macro_fields(&d_fields);

    return 0;
}