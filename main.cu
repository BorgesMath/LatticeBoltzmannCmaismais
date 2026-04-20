#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <string>
#include <clocale>

#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "multiphase/cahn_hilliard.cuh"
#include "Magnetismo/poisson.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"
#include "lbm/lbm.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Erro CUDA na linha " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void allocate_populations(LBM_Populations* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->f0), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f2), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f4), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f6), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f8), bytes));
}

void allocate_macro_fields(Macro_Fields* f, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(f->phi), bytes));      CUDA_CHECK(cudaMalloc((void**)&(f->phi_new), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->mu), bytes));       CUDA_CHECK(cudaMalloc((void**)&(f->psi), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->rho), bytes));      CUDA_CHECK(cudaMalloc((void**)&(f->ux), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->uy), bytes));       CUDA_CHECK(cudaMalloc((void**)&(f->chi_field), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(f->K_field), bytes));
}

void free_populations(LBM_Populations* p) {
    cudaFree(p->f0); cudaFree(p->f1); cudaFree(p->f2); cudaFree(p->f3);
    cudaFree(p->f4); cudaFree(p->f5); cudaFree(p->f6); cudaFree(p->f7); cudaFree(p->f8);
}

void free_macro_fields(Macro_Fields* f) {
    cudaFree(f->phi); cudaFree(f->phi_new); cudaFree(f->mu); cudaFree(f->psi);
    cudaFree(f->rho); cudaFree(f->ux); cudaFree(f->uy); cudaFree(f->chi_field); cudaFree(f->K_field);
}

void swap_populations(LBM_Populations* p1, LBM_Populations* p2) {
    LBM_Populations temp = *p1; *p1 = *p2; *p2 = temp;
}

void print_progress_bar(int step, int total) {
    float progress = (float)step / total;
    int barWidth = 40;
    std::cout << "\rProgress: [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << step << "/" << total << ")";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // Força o padrão decimal internacional (ponto) para std::stod
    std::setlocale(LC_NUMERIC, "C");

    // 1. Configuração do Raio da Gota via Terminal
    double R0_val = 20.0;
    if (argc >= 2) {
        R0_val = std::stod(argv[1]);
    }

    // 2. Preparação de Memória
    LBM_Populations d_f_in, d_f_out;
    Macro_Fields d_fields;
    size_t mem_size = NUM_NODES * sizeof(double);

    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    double *h_phi = (double*)malloc(mem_size);
    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    // Inicialização do diretório e gravação de metadados para automação
    std::string output_dir = init_post_processing();
    save_metadata_laplace(output_dir, R0_val);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 3. Inicialização da Gota
    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields, R0_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cópia inicial das populações
    CUDA_CHECK(cudaMemcpy(d_f_out.f0, d_f_in.f0, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f1, d_f_in.f1, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f2, d_f_in.f2, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f3, d_f_in.f3, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f4, d_f_in.f4, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f5, d_f_in.f5, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f6, d_f_in.f6, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f7, d_f_in.f7, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f8, d_f_in.f8, mem_size, cudaMemcpyDeviceToDevice));

    int max_iter = 10000; // Tempo suficiente para estabilização de Laplace
    double chi_max = 1.2;

    std::cout << "-> Iniciando Validacao de Laplace | R0 = " << R0_val << " | " << output_dir << std::endl;

    // 4. Ciclo Principal da Simulação
    for (int t = 0; t <= max_iter; ++t) {

        // ACURÁCIA TEMPORAL: Sub-passos de Cahn-Hilliard para estabilizar a fase
        for (int sub = 0; sub < CH_SUBSTEPS; ++sub) {
            solve_cahn_hilliard(d_fields, numBlocks, threadsPerBlock);
        }

        // Acoplamento Suscetibilidade e Magnético
        update_susceptibility_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, chi_max);
        CUDA_CHECK(cudaDeviceSynchronize());
        solve_poisson_magnetic(d_fields, numBlocks, threadsPerBlock);

        // Colisão e Propagação LBM
        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Correção das fronteiras abertas (apenas 3 argumentos exigidos pelo multifásico)
        if (!IS_PERIODIC) {
            apply_open_boundaries(d_f_out, d_fields, NY);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        swap_populations(&d_f_in, &d_f_out);

        // Exportação VTK e Barra de Progresso
        if (t % SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_phi, h_rho, h_ux, h_uy);
        }
        if (t % 200 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter);
        }
    }

    std::cout << "\nSimulacao concluida." << std::endl;

    // 5. Limpeza de Recursos
    free(h_phi); free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out);
    free_macro_fields(&d_fields);

    return 0;
}