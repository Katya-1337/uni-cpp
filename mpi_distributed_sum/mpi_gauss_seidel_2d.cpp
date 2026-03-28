#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// Индексация локального блока: [i][j] → i * local_M + j
inline int idx(int i, int j, int stride) {
    return i * stride + j;
}

double solve_gauss_seidel_mpi_2d(int N, double eps = 1e-4) {
    const double h = 1.0 / (N + 1.0);
    const double h2 = h * h;
    const int M = N + 2; // глобальная ширина (включая границы)

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // === 1. Создаём 2D декартову топологию ===
    int dims[2] = { 0, 0 };
    int periods[2] = { 0, 0 }; // не периодические
    int reorder = 1;

    // Пытаемся подобрать dims[0] × dims[1] = world_size, близкие к квадрату
    dims[0] = static_cast<int>(sqrt(world_size));
    while (dims[0] > 0 && world_size % dims[0] != 0) dims[0]--;
    if (dims[0] == 0) dims[0] = 1;
    dims[1] = world_size / dims[0];

    // Можно также явно задать, например: dims[0]=2, dims[1]=4 для 8 процессов
    // Но автоматический подбор — удобнее

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int cart_rank;
    int coords[2];
    int dims_out[2];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    MPI_Cart_get(cart_comm, 2, dims_out, periods, periods);
    int Px = dims_out[0], Py = dims_out[1]; // Px × Py = world_size
    int px = coords[0], py = coords[1];    // мой номер по строкам/столбцам

    // === 2. Делим N строк и N столбцов между процессами ===
    // Количество строк на процесс (внутренние, без границ)
    int rows_per_proc = N / Px;
    int extra_rows = N % Px;
    int local_Nx = rows_per_proc + (px < extra_rows ? 1 : 0);
    int start_i = px * rows_per_proc + (px < extra_rows ? px : extra_rows);

    // Количество столбцов на процесс
    int cols_per_proc = N / Py;
    int extra_cols = N % Py;
    int local_Ny = cols_per_proc + (py < extra_cols ? 1 : 0);
    int start_j = py * cols_per_proc + (py < extra_cols ? py : extra_cols);

    // === 3. Выделяем локальный блок: (local_Nx + 2) × (local_Ny + 2)
    // +2 — ghost-слои: сверху/снизу и слева/справа
    int local_M = local_Ny + 2;
    std::vector<double> u((local_Nx + 2) * local_M, 0.0);

    // === 4. Соседи по 4 направлениям ===
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // ось 0 — строки (i)
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // ось 1 — столбцы (j)

    // === 5. Время начала ===
    double start_time = MPI_Wtime();

    // === 6. Итерации ===
    double dmax_global = 1.0;
    int iters = 0;

    while (dmax_global > eps) {
        double dmax_local = 0.0;

        // --- Обмен ghost-слоями ---
        // Обмен по вертикали (строки i=1 и i=local_Nx)
        {
            MPI_Request reqs[4];
            int nreq = 0;

            // Получить строку сверху → в u[0][*]
            if (north != MPI_PROC_NULL) {
                MPI_Irecv(&u[idx(0, 1, local_M)], local_Ny, MPI_DOUBLE, north, 1, cart_comm, &reqs[nreq++]);
            }
            // Получить строку снизу → в u[local_Nx+1][*]
            if (south != MPI_PROC_NULL) {
                MPI_Irecv(&u[idx(local_Nx + 1, 1, local_M)], local_Ny, MPI_DOUBLE, south, 0, cart_comm, &reqs[nreq++]);
            }
            // Отправить свою первую строку (i=1) вверх
            if (north != MPI_PROC_NULL) {
                MPI_Isend(&u[idx(1, 1, local_M)], local_Ny, MPI_DOUBLE, north, 0, cart_comm, &reqs[nreq++]);
            }
            // Отправить свою последнюю строку (i=local_Nx) вниз
            if (south != MPI_PROC_NULL) {
                MPI_Isend(&u[idx(local_Nx, 1, local_M)], local_Ny, MPI_DOUBLE, south, 1, cart_comm, &reqs[nreq++]);
            }
            if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Обмен по горизонтали (столбцы j=1 и j=local_Ny)
        {
            MPI_Request reqs[4];
            int nreq = 0;

            // Получить столбец слева → в u[*][0]
            if (west != MPI_PROC_NULL) {
                for (int i = 1; i <= local_Nx; i++) {
                    MPI_Irecv(&u[idx(i, 0, local_M)], 1, MPI_DOUBLE, west, 1, cart_comm, &reqs[nreq++]);
                }
            }
            // Получить столбец справа → в u[*][local_Ny+1]
            if (east != MPI_PROC_NULL) {
                for (int i = 1; i <= local_Nx; i++) {
                    MPI_Irecv(&u[idx(i, local_Ny + 1, local_M)], 1, MPI_DOUBLE, east, 0, cart_comm, &reqs[nreq++]);
                }
            }
            // Отправить свой первый столбец (j=1) налево
            if (west != MPI_PROC_NULL) {
                for (int i = 1; i <= local_Nx; i++) {
                    MPI_Isend(&u[idx(i, 1, local_M)], 1, MPI_DOUBLE, west, 0, cart_comm, &reqs[nreq++]);
                }
            }
            // Отправить свой последний столбец (j=local_Ny) направо
            if (east != MPI_PROC_NULL) {
                for (int i = 1; i <= local_Nx; i++) {
                    MPI_Isend(&u[idx(i, local_Ny, local_M)], 1, MPI_DOUBLE, east, 1, cart_comm, &reqs[nreq++]);
                }
            }
            if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // --- Red-Black обновление ---
        for (int color = 0; color < 2; color++) {
            for (int li = 1; li <= local_Nx; li++) {
                int gi = start_i + li;  // глобальный i: 1..N
                for (int lj = 1; lj <= local_Ny; lj++) {
                    int gj = start_j + lj;  // глобальный j: 1..N
                    if (((gi + gj) & 1) != color) continue;

                    int k = idx(li, lj, local_M);
                    double old = u[k];
                    double newv = 0.25 * (
                        u[idx(li - 1, lj, local_M)] +  // север
                        u[idx(li + 1, lj, local_M)] +  // юг
                        u[idx(li, lj - 1, local_M)] +  // запад
                        u[idx(li, lj + 1, local_M)] -  // восток
                        h2 * 1.0
                        );
                    u[k] = newv;
                    double diff = std::abs(newv - old);
                    if (diff > dmax_local) dmax_local = diff;
                }
            }
        }

        // Собираем глобальный максимум
        MPI_Allreduce(&dmax_local, &dmax_global, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        iters++;
    }

    double end_time = MPI_Wtime();
    MPI_Comm_free(&cart_comm);
    return end_time - start_time;
}

// Вспомогательная функция: получить T1 (serial reference)
double get_serial_time(int N, double eps = 1e-4) {
    // Имитируем serial через один процесс в MPI — но проще сделать отдельную функцию без MPI
    // Для корректного speedup нужен настоящий serial (без overhead MPI)
    // Здесь — приближённо: запускаем solve_gauss_seidel_mpi_2d в 1 процессе
    // Но так как MPI инициализирован, просто вернём время одного процесса при world_size=1
    // В реальности лучше вынести serial в отдельную функцию (как в вашем OpenMP-коде)
    // Пока для простоты — пропустим и посчитаем speedup относительно 1 процесса
    return 0.0; // placeholder
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> Ns = { 10, 100, 1000, 2000, 3000 };
    std::vector<double> times;

    if (rank == 0) {
        std::cout << "N\tProcs\tTime(s)\n";
        std::cout << std::fixed << std::setprecision(4);
    }

    for (int N : Ns) {
        double time = solve_gauss_seidel_mpi_2d(N);
        times.push_back(time);

        if (rank == 0) {
            std::cout << N << "\t" << size << "\t" << time << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // === Оценка speedup (если запускали с разными size) ===
    // Чтобы получить speedup, нужно запускать разные mpiexec -np 1, 2, 4, ...
    // И сохранять T1 отдельно.

    // Например, можно сохранить времена в файл:
    if (rank == 0) {
        std::string fname = "mpi_times_p" + std::to_string(size) + ".txt";
        std::ofstream f(fname);
        if (f.is_open()) {
            f << "N,Time\n";
            for (size_t i = 0; i < Ns.size(); i++) {
                f << Ns[i] << "," << times[i] << "\n";
            }
            f.close();
            std::cout << "→ Saved to " << fname << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}