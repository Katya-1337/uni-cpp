// Метод Гаусса–Зейделя для эллиптического уравнения, 1D-раскладка (MPI).

#include "mpi.h"
#include <iostream>
#include <vector>
#include <cmath>

inline int idx(int i, int j, int M) {
    return i * M + j;
}

// Решение Гаусса–Зейделя с помощью MPI
double solve_gauss_seidel_mpi(int N, double eps = 1e-4) {
    const double h = 1.0 / (N + 1.0);
    const double h2 = h * h;
    const int M = N + 2;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Разделяем строки между процессами (N строк внутри домена, индексы 1..N)
    int local_N = N / size;
    int remainder = N % size;
    // Балансировка: первые 'remainder' процессов получат на 1 строку больше
    if (rank < remainder) {
        local_N++;
    }

    // Вычислим смещение (начальный глобальный индекс строки i=1 соответствует rank=0)
    int start_i = 1;
    for (int r = 0; r < rank; r++) {
        int rows_r = N / size + (r < remainder ? 1 : 0);
        start_i += rows_r;
    }

    // Локальный блок: строк local_N + 2 (ghost rows сверху и снизу)
    int local_M = local_N + 2;
    std::vector<double> u(local_M * M, 0.0);

    // Время начала
    double start = MPI_Wtime();

    int iters = 0;
    double dmax_global = 1.0;

    // Типы данных для обмена — строки
    MPI_Datatype row_type;
    MPI_Type_contiguous(M, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    while (dmax_global > eps) {
        double dmax_local = 0.0;

        // Обмен граничными строками: u[1] (верх) и u[local_N] (низ) → соседям
        const int UP = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        const int DOWN = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // Обмен ghost rows: u[1] ← от UP, u[local_N] ← от DOWN
        // Отправляем свою внутреннюю верхнюю строку (индекс 1) вверх → становится ghost в UP (внизу его блока: local_M-1)
        // И свою внутреннюю нижнюю строку (индекс local_N) вниз → становится ghost в DOWN (вверху его блока: 0)
        MPI_Request reqs[4];
        int req_count = 0;

        // Receive top ghost (into u[0])
        if (UP != MPI_PROC_NULL) {
            MPI_Irecv(&u[idx(0, 0, M)], 1, row_type, UP, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Receive bottom ghost (into u[local_N+1])
        if (DOWN != MPI_PROC_NULL) {
            MPI_Irecv(&u[idx(local_N + 1, 0, M)], 1, row_type, DOWN, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Send top internal row (u[1]) to UP
        if (UP != MPI_PROC_NULL) {
            MPI_Isend(&u[idx(1, 0, M)], 1, row_type, UP, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // Send bottom internal row (u[local_N]) to DOWN
        if (DOWN != MPI_PROC_NULL) {
            MPI_Isend(&u[idx(local_N, 0, M)], 1, row_type, DOWN, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // Ждём все обмены
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        // Два цвета: чётность (i+j) % 2
        for (int color = 0; color < 2; color++) {
            // Обходим только свои строки: локальные индексы 1..local_N соответствуют глобальным start_i .. start_i+local_N-1
            for (int li = 1; li <= local_N; li++) {
                int gi = start_i + li - 1;  // глобальный i
                for (int j = 1; j <= N; j++) {
                    if (((gi + j) & 1) != color) continue;

                    int k = idx(li, j, M);
                    double old = u[k];
                    double newv = 0.25 * (
                        u[idx(li - 1, j, M)] +   // i-1 (может быть ghost)
                        u[idx(li + 1, j, M)] +   // i+1 (может быть ghost)
                        u[idx(li, j - 1, M)] +
                        u[idx(li, j + 1, M)] -
                        h2 * 1.0
                        );

                    u[k] = newv;
                    double d = std::abs(newv - old);
                    if (d > dmax_local) dmax_local = d;
                }
            }
        }

        // Собираем глобальный максимум
        MPI_Allreduce(&dmax_local, &dmax_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iters++;
    }

    MPI_Type_free(&row_type);
    double end = MPI_Wtime();
    return end - start;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Чтобы не дублировать вывод, пусть только rank 0 печатает
    if (rank == 0) {
        std::cout << "N\tProcs\tTime(s)\n";
    }

    std::vector<int> Ns = { 10, 100, 1000, 2000, 3000}; // для больших N убедитесь, что N >= size!

    for (int N : Ns) {
        if (N < size) {
            if (rank == 0) {
                std::cout << N << "\t" << size << "\tN too small for " << size << " procs — skipped\n";
            }
            continue;
        }

        double time = solve_gauss_seidel_mpi(N);

        if (rank == 0) {
            std::cout << N << "\t" << size << "\t" << time << "\n";
        }
        // Синхронизируем, чтобы не было пересечения выводов
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}