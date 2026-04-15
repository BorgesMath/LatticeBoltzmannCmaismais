#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <string>
#include <clocale> // Necessário para lidar com a formatação decimal de diferentes regiões

#include "config/config.cuh"
#include "initialization/initialization.cuh"
#include "boundaries/open_boundaries.cuh"
#include "post_process/post_process.cuh"
#include "lbm/lbm.cuh"

// Macro para verificação de erros CUDA
#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "Erro CUDA em " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while (0)

// Alocação das populações do reticulado
void allocate_populations(LBM_Populations* p, size_t bytes) {
    CUDA_CHECK(cudaMalloc((void**)&(p->f0), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f1), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f2), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f3), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f4), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f5), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f6), bytes)); CUDA_CHECK(cudaMalloc((void**)&(p->f7), bytes));
    CUDA_CHECK(cudaMalloc((void**)&(p->f8), bytes));
}

// Alocação dos campos macroscópicos
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
    // Força o padrão decimal "C" (ponto em vez de vírgula) para evitar erros no std::stod
    std::setlocale(LC_NUMERIC, "C");

    // 1. Configuração de Parâmetros Dinâmicos via Command Line
    double K_0_val = 20.0;     // Valor padrão
    double U_inlet_val = 0.01; // Valor padrão

    if (argc >= 3) {
        K_0_val = std::stod(argv[1]);     // Argumento 1: Permeabilidade (ex: 50.0)
        U_inlet_val = std::stod(argv[2]); // Argumento 2: Velocidade de entrada (ex: 0.005)
    }

    // 2. Preparação de Memória
    LBM_Populations d_f_in, d_f_out;
    Macro_Fields d_fields;
    size_t mem_size = NUM_NODES * sizeof(double);

    allocate_populations(&d_f_in, mem_size);
    allocate_populations(&d_f_out, mem_size);
    allocate_macro_fields(&d_fields, mem_size);

    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    // Inicializa o diretório e salva os metadados para o Python ler depois
    std::string output_dir = init_post_processing();
    save_metadata(output_dir, K_0_val, U_inlet_val);

    // 3. Configuração da Grid GPU
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. Inicialização Dinâmica
    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields, K_0_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    int max_iter = 50000;
    std::cout << "======================================================" << std::endl;
    std::cout << " LBM SOLVER - VALIDAÇÃO PARAMÉTRICA" << std::endl;
    std::cout << " Domínio: " << NX << "x" << NY << " | Iterações: " << max_iter << std::endl;
    std::cout << " Parâmetros -> K_0: " << K_0_val << " | U_inlet: " << U_inlet_val << std::endl;
    std::cout << " Diretório: " << output_dir << std::endl;
    std::cout << "======================================================" << std::endl;

    // 5. Ciclo Principal
    for (int t = 0; t <= max_iter; ++t) {

        // Colisão e Propagação
        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Fronteira Dinâmica
        apply_open_boundaries(d_f_out, d_fields, NY, U_inlet_val);
        CUDA_CHECK(cudaDeviceSynchronize());

        swap_populations(&d_f_in, &d_f_out);

        // Exportação VTK
        if (t % SNAPSHOT_STEPS == 0) {
            export_vtk(t, output_dir, d_fields, h_rho, h_ux, h_uy);
        }

        if (t % 200 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter);
        }
    }

    std::cout << "\nSimulação concluída." << std::endl;

    // 6. Limpeza
    free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out);
    free_macro_fields(&d_fields);

    return 0;
}