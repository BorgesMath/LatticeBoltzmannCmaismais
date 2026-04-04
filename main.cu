#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

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
    LBM_Populations temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

void check_numerical_stability() {
    std::cout << "========================================" << std::endl;
    std::cout << "DIAGNOSTICO DE ESTABILIDADE LBM" << std::endl;
    std::cout << "========================================" << std::endl;

    double cs = 1.0 / sqrt(3.0);
    double mach = U_INLET / cs;
    std::cout << "Numero de Mach (Inlet): " << mach;
    if (mach > 0.1) std::cout << " [ALERTA: Risco de compressibilidade]" << std::endl;
    else std::cout << " [OK]" << std::endl;

    double cfl_ch = (M_MOBILITY * DT_CH) / 1.0;
    std::cout << "Numero CFL (Cahn-Hilliard): " << cfl_ch;
    if (cfl_ch > 0.1) std::cout << " [ALERTA: Integracao pode divergir]" << std::endl;
    else std::cout << " [OK]" << std::endl;

    if (IS_PERIODIC) std::cout << "\n[MODO]: Validacao Monofasica/Laplace (Periodico)." << std::endl;
    else std::cout << "\n[MODO]: Producao Saffman-Taylor (Aberto)." << std::endl;

    std::cout << "========================================\n" << std::endl;
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

    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% "
              << "| It: " << std::setw(5) << step << " / " << total << "     ";
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

    double *h_phi = (double*)malloc(mem_size);
    double *h_rho = (double*)malloc(mem_size);
    double *h_ux  = (double*)malloc(mem_size);
    double *h_uy  = (double*)malloc(mem_size);

    if (!h_phi || !h_rho || !h_ux || !h_uy) {
        std::cerr << "Falha de segmentacao na alocacao de RAM do Host." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string output_dir = init_post_processing();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 1. Injeção da Condição Inicial (Gota Analítica ou Saffman-Taylor)
    init_fields_kernel<<<numBlocks, threadsPerBlock>>>(d_f_in, d_fields);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sincronização inicial de distribuições
    CUDA_CHECK(cudaMemcpy(d_f_out.f0, d_f_in.f0, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f1, d_f_in.f1, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f2, d_f_in.f2, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f3, d_f_in.f3, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f4, d_f_in.f4, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f5, d_f_in.f5, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f6, d_f_in.f6, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f7, d_f_in.f7, mem_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_out.f8, d_f_in.f8, mem_size, cudaMemcpyDeviceToDevice));

    check_numerical_stability();

    // =========================================================================
    // 2. PRÉ-CONDICIONAMENTO TERMODINÂMICO (WARM-UP)
    // Minimiza a energia livre de Ginzburg-Landau sem acionar o fluido,
    // garantindo gradiente nulo do potencial químico e anulação do choque inicial.
    // =========================================================================
    std::cout << "Executando Pre-Condicionamento Termodinamico (2500 steps)..." << std::endl;
    for (int p = 0; p < 2500; ++p) {
        solve_cahn_hilliard(d_fields, numBlocks, threadsPerBlock);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Pre-Condicionamento concluido. Interface estabilizada." << std::endl;
    // =========================================================================

    // Ajuste de horizonte temporal para garantir relaxação de correntes espúrias
    int max_iter = IS_PERIODIC ? 40000 : 20000;
    double chi_max = 1.2;

    // 3. INTEGRAÇÃO TEMPORAL ACOPLADA
    for (int t = 0; t <= max_iter; ++t) {

        // Evolução de Fase
        solve_cahn_hilliard(d_fields, numBlocks, threadsPerBlock);

        // Acoplamento Magnético (Somente se ativo no config)
        update_susceptibility_kernel<<<numBlocks, threadsPerBlock>>>(d_fields, chi_max);
        CUDA_CHECK(cudaDeviceSynchronize());
        solve_poisson_magnetic(d_fields, numBlocks, threadsPerBlock);

        // Resolução Hidrodinâmica via Lattice Boltzmann
        lbm_collide_and_stream<<<numBlocks, threadsPerBlock>>>(d_f_in, d_f_out, d_fields);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Condições de Contorno de Escoamento (Desativado no teste de Laplace)
        if (!IS_PERIODIC) {
            apply_open_boundaries(d_f_out, d_fields, NY);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        swap_populations(&d_f_in, &d_f_out);

        // Exportação de Dados e Snapshot
        if (t % SNAPSHOT_STEPS == 0 && t > 0) {
            export_vtk(t, output_dir, d_fields, h_phi, h_rho, h_ux, h_uy);
        }

        if (t % 100 == 0 || t == max_iter) {
            print_progress_bar(t, max_iter);
        }
    }

    // Liberação de Memória
    free(h_phi); free(h_rho); free(h_ux); free(h_uy);
    free_populations(&d_f_in); free_populations(&d_f_out); free_macro_fields(&d_fields);

    return 0;
}