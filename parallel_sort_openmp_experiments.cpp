#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>

int main() {
    // Размеры для тестирования
    int sizes[] = { 100000, 500000, 1000000, 5000000, 10000000 };
    int threads[] = { 1, 2, 3, 6 };

    std::cout << "ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА (parallel for)\n";
    std::cout << "=================================================\n";
    std::cout << "Размер\t\tПослед.\t\t2 пот.\t\t3 пот.\t\t6 поток.\n";
    std::cout << "-------------------------------------------------\n";

    for (int size : sizes) {
        std::cout << size << "\t\t";

        for (int num_threads : threads) {
            // Генерация массива
            std::vector<int> arr(size);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 1000000);

            for (int i = 0; i < size; i++) arr[i] = dis(gen);

            // Установка потоков
            omp_set_num_threads(num_threads);

            auto start = std::chrono::high_resolution_clock::now();

            if (num_threads == 1) {
                std::sort(arr.begin(), arr.end());
            }
            else {
                // Разделяем массив на части и сортируем каждую часть параллельно
#pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();
                    int thread_count = omp_get_num_threads();
                    int chunk_size = size / thread_count;
                    int start_idx = thread_id * chunk_size;
                    int end_idx = (thread_id == thread_count - 1) ? size : start_idx + chunk_size;

                    std::sort(arr.begin() + start_idx, arr.begin() + end_idx);
                }

                // Затем сливаем все части
                for (int step = 1; step < num_threads; step *= 2) {
#pragma omp parallel for
                    for (int i = 0; i < num_threads; i += 2 * step) {
                        if (i + step < num_threads) {
                            int start1 = i * (size / num_threads);
                            int end1 = (i + step) * (size / num_threads);
                            int end2 = (i + 2 * step) * (size / num_threads);
                            if (end2 > size) end2 = size;

                            std::inplace_merge(
                                arr.begin() + start1,
                                arr.begin() + end1,
                                arr.begin() + end2
                            );
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();

            std::cout << time << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}