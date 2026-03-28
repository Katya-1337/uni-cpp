#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

// Последовательная сортировка (std::sort)
double sequential_sort(std::vector<int>& arr) {
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(arr.begin(), arr.end());
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// Параллельный QuickSort с OpenMP
// min_size — порог, ниже которого используем std::sort (во избежание накладных)
void parallel_qsort(std::vector<int>& arr, int left, int right, int depth = 0) {
    const int MIN_SIZE = 10000; // порог переключения
    const int MAX_DEPTH = 4;    // ограничение вложенности параллелизма (log2(16)=4 → до 16 активных задач)

    while (right > left) {
        if (right - left < MIN_SIZE) {
            std::sort(arr.begin() + left, arr.begin() + right + 1);
            return;
        }

        // Разделение (Lomuto или Hoare — здесь Hoare для баланса)
        int i = left, j = right;
        int pivot = arr[left + (right - left) / 2]; // медиана трёх можно, но не критично

        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                std::swap(arr[i], arr[j]);
                i++; j--;
            }
        }

        // Рекурсивно сортируем меньшую часть в цикле, большую — параллельно или рекурсивно
        bool left_is_smaller = (j - left) < (right - i);

        if (depth < MAX_DEPTH && omp_get_active_level() < omp_get_max_active_levels()) {
            // Параллельно: одна ветвь в новом потоке, другая — последовательно
#pragma omp task if(depth < MAX_DEPTH) final(depth >= MAX_DEPTH - 1)
            {
                if (left_is_smaller && left < j)
                    parallel_qsort(arr, left, j, depth + 1);
                if (!left_is_smaller && i < right)
                    parallel_qsort(arr, i, right, depth + 1);
            }
            // текущий поток делает противоположную ветвь
            if (left_is_smaller) {
                left = i;
            }
            else {
                right = j;
            }
        }
        else {
            // Последовательно: сначала меньшую часть
            if (left_is_smaller) {
                if (left < j) parallel_qsort(arr, left, j, depth + 1);
                left = i;
            }
            else {
                if (i < right) parallel_qsort(arr, i, right, depth + 1);
                right = j;
            }
        }
    }
}

double parallel_sort(std::vector<int>& arr, int num_threads) {
    omp_set_num_threads(num_threads);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
#pragma omp single
    {
        parallel_qsort(arr, 0, static_cast<int>(arr.size()) - 1, 0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// Генерация случайного массива
std::vector<int> generate_random_array(size_t n) {
    std::vector<int> arr(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000000);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

int main() {
    std::vector<size_t> sizes = { 100000, 500000, 1000000, 5000000, 10000000, 50000000 };
    std::vector<int> thread_counts = { 2, 4, 8 };

    std::cout << "Количество элементов\t"
        << "Последовательный (сек)\t"
        << "2 процессора: Время\tУскорение\t"
        << "4 процессора: Время\tУскорение\t"
        << "8 процессоров: Время\tУскорение\n";

    for (size_t n : sizes) {
        // Последовательная сортировка — один раз (для справедливости)
        auto arr_seq = generate_random_array(n);
        double t_seq = sequential_sort(arr_seq);

        // Ускорения
        std::vector<double> t_par(thread_counts.size());
        std::vector<double> speedup(thread_counts.size());

        for (size_t ti = 0; ti < thread_counts.size(); ++ti) {
            int threads = thread_counts[ti];
            auto arr_par = generate_random_array(n); // новый случайный массив
            t_par[ti] = parallel_sort(arr_par, threads);
            speedup[ti] = t_seq / t_par[ti];
        }

        std::cout << n << "\t"
            << t_seq << "\t"
            << t_par[0] << "\t" << speedup[0] << "\t"
            << t_par[1] << "\t" << speedup[1] << "\t"
            << t_par[2] << "\t" << speedup[2] << "\n";
    }

    return 0;
}