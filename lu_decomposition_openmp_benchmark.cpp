#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;

// Умножение матриц L (n x n) и U (n x n) -> результат в res
void multiplyMatrices(const double* L, const double* U, double* res, int n) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                res[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }
}


// LU-разложение по методу Дулиттла (L с единичной диагональю)
void luDecomposition(double* A, double* L, double* U, int n) {
    // Инициализация L и U нулями
#pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        // Вычисление строки U[i][j], j = i..n-1
#pragma omp parallel for
        for (int j = i; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[i * n + k] * U[k * n + j];
            }
            U[i * n + j] = A[i * n + j] - sum;
        }

        // Вычисление столбца L[j][i], j = i..n-1
#pragma omp parallel for
        for (int j = i; j < n; ++j) {
            if (i == j) {
                L[i * n + i] = 1.0; // диагональ L = 1
            }
            else {
                double sum = 0.0;
                for (int k = 0; k < i; ++k) {
                    sum += L[j * n + k] * U[k * n + i];
                }
                L[j * n + i] = (A[j * n + i] - sum) / U[i * n + i];
            }
        }
    }
}

// Решение Ly = b (прямой ход)
void forwardSubstitution(const double* L, const double* b, double* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L[i * n + j] * y[j];
        }
    }
}

// Решение Ux = y (обратный ход)
void backwardSubstitution(const double* U, const double* y, double* x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= U[i * n + j] * x[j];
        }
        x[i] /= U[i * n + i];
    }
}

// Генерация случайной матрицы и вектора
void generateRandomSystem(double* A, double* b, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.1, 10.0);

    // Генерация диагонально доминирующей матрицы для устойчивости
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                A[i * n + j] = dist(gen) + n; // диагональное преобладание
            }
            else {
                A[i * n + j] = dist(gen);
            }
        }
    }

    // Генерация вектора b
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        b[i] = dist(gen) * 10.0;
    }
}

// Измерение времени выполнения LU-разложения
double measureLUPerformance(double* A, double* L, double* U, int n, int threads) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    auto start = high_resolution_clock::now();
    luDecomposition(A, L, U, n);
    auto end = high_resolution_clock::now();

    return duration_cast<duration<double>>(end - start).count();
}

int main() {
    setlocale(LC_ALL, "russian");

    const int n = 1000; // Размер матрицы 1000x1000
    vector<int> threads_list = { 1, 2, 4, 8 }; // Тестируемые количества потоков

#ifdef _OPENMP
    cout << "Поддержка OpenMP включена" << endl;
#else
    cout << "OpenMP не поддерживается" << endl;
#endif

    // Выделение памяти
    double* A = new double[n * n];
    double* b = new double[n];
    double* L = new double[n * n];
    double* U = new double[n * n];
    double* y = new double[n];
    double* x = new double[n];

    cout << "Генерация случайной системы " << n << "x" << n << "..." << endl;
    generateRandomSystem(A, b, n);

    vector<double> execution_times;
    vector<double> speedups;
    vector<double> efficiencies;

    // Измерение производительности для разного количества потоков
    for (int threads : threads_list) {
        cout << "\nТестирование с " << threads << " потоком(ами)..." << endl;

        // Копирование исходных данных для каждого теста
        double* A_copy = new double[n * n];
        double* L_copy = new double[n * n];
        double* U_copy = new double[n * n];

        copy(A, A + n * n, A_copy);

        double time = measureLUPerformance(A_copy, L_copy, U_copy, n, threads);
        execution_times.push_back(time);

        cout << "Время выполнения: " << time << " секунд" << endl;

        delete[] A_copy;
        delete[] L_copy;
        delete[] U_copy;
    }

    // Расчет ускорения и эффективности
    double serial_time = execution_times[0]; // Время для 1 потока

    // Вывод заголовка таблицы
    cout << "\n" << string(60, '=') << endl;
    cout << "РЕЗУЛЬТАТЫ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ В EXCEL" << endl;
    cout << string(60, '=') << endl;
    cout << "Потоки\tВремя(с)\tУскорение\tЭффективность(%)" << endl;
    cout << string(60, '-') << endl;

    for (size_t i = 0; i < execution_times.size(); ++i) {
        double speedup = serial_time / execution_times[i];
        double efficiency = (speedup / threads_list[i]) * 100;

        speedups.push_back(speedup);
        efficiencies.push_back(efficiency);

        // Вывод данных в табличном формате для копирования в Excel
        cout << threads_list[i] << "\t"
            << fixed << setprecision(4) << execution_times[i] << "\t\t"
            << fixed << setprecision(4) << speedup << "\t\t"
            << fixed << setprecision(2) << efficiency << endl;
    }

    // Проверка корректности решения с максимальным количеством потоков
    cout << "\n" << string(60, '=') << endl;
    cout << "ПРОВЕРКА РЕШЕНИЯ СИСТЕМЫ:" << endl;
    cout << string(60, '=') << endl;

#ifdef _OPENMP
    omp_set_num_threads(threads_list.back());
#endif

    auto start = high_resolution_clock::now();

    luDecomposition(A, L, U, n);
    forwardSubstitution(L, b, y, n);
    backwardSubstitution(U, y, x, n);

    auto end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end - start).count();

    cout << "Полное время решения системы: " << total_time << " секунд" << endl;

    // Проверка решения (вычисление невязки)
    double max_residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double Ax = 0.0;
        for (int j = 0; j < n; ++j) {
            Ax += A[i * n + j] * x[j];
        }
        double residual = fabs(Ax - b[i]);
        if (residual > max_residual) {
            max_residual = residual;
        }
    }
    cout << "Максимальная невязка: " << max_residual << endl;
    cout << "Система решена корректно!" << endl;

    // Освобождение памяти
    delete[] A;
    delete[] b;
    delete[] L;
    delete[] U;
    delete[] y;
    delete[] x;

    return 0;
}