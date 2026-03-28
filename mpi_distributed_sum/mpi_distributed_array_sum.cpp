// Распределённое суммирование элементов массива (MPI).

#include <iostream>
#include "mpi.h"

#define N 120000

int main(int argc, char* argv[]) {
    double* x = nullptr;
    double local_sum = 0.0, total_sum = 0.0;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Синхронизация и запуск таймера
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Рассылаем массив всем
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределяем работу
    int elements_per_proc = N / size;
    int remainder = N % size;

    int start, end;
    if (rank < remainder) {
        start = rank * (elements_per_proc + 1);
        end = start + elements_per_proc + 1;
    }
    else {
        start = remainder * (elements_per_proc + 1) + (rank - remainder) * elements_per_proc;
        end = start + elements_per_proc;
    }

    // Суммируем свою часть
    for (int i = start; i < end; i++) {
        local_sum += x[i];
    }

    // Собираем результат
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        double exec_time = end_time - start_time;
        std::cout << "Processes: " << size
            << " | Time: " << exec_time << " seconds"
            << " | Total Sum: " << total_sum << std::endl;
    }

    delete[] x;
    MPI_Finalize();
    return 0;
}