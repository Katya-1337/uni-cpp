#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;

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
        // L[i][i] = 1, делить не нужно
    }
}

// Решение Ux = y (обратный ход)
void backwardSubstitution(const double* U, const double* y, double* x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= U[i * n + j] * x[j];
        }
        if (U[i * n + i] == 0.0) {
            cerr << "Матрица вырождена: деление на ноль при обратной подстановке!" << endl;
            return;
        }
        x[i] /= U[i * n + i];
    }
}

// Печать матрицы
void printMatrix(const double* M, int n, const string& name) {
    cout << name << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << setw(10) << fixed << setprecision(4) << M[i * n + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

// Печать вектора
void printVector(const double* v, int n, const string& name) {
    cout << name << ":\n";
    for (int i = 0; i < n; ++i) {
        cout << setw(10) << fixed << setprecision(4) << v[i] << "\n";
    }
    cout << "\n";
}

// Сравнение двух матриц (вывод разности)
void compareMatrices(const double* A, const double* A_rec, int n) {
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = fabs(A[i * n + j] - A_rec[i * n + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    cout << "Максимальная разница между A и L*U: " << max_diff << endl;
}

int main() {
#ifdef _OPENMP
    setlocale(LC_ALL, "russian");
    cout << "Поддержка OpenMP включена" << endl;

    int num_threads;
    cout << "Введите количество потоков: ";
    cin >> num_threads;
    omp_set_num_threads(num_threads);
#else
    cout << "OpenMP не поддерживается" << endl;
#endif

    int n;
    cout << "Введите размерность системы (n): ";
    cin >> n;

    if (n <= 0) {
        cerr << "Некорректная размерность!" << endl;
        return 1;
    }

    double* A = new double[n * n];
    double* b = new double[n];
    double* L = new double[n * n];
    double* U = new double[n * n];
    double* A_reconstructed = new double[n * n];
    double* y = new double[n];
    double* x = new double[n];

    // Ввод матрицы A
    cout << "Введите коэффициенты матрицы A (" << n << "x" << n << "):" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << "a[" << i << "][" << j << "] = ";
            cin >> A[i * n + j];
        }
    }

    // Ввод вектора b
    cout << "Введите свободные члены вектора b (" << n << " элементов):" << endl;
    for (int i = 0; i < n; i++) {
        cout << "b[" << i << "] = ";
        cin >> b[i];
    }

    // 1) LU-разложение
    cout << "\nВыполняется LU-разложение...\n";
    luDecomposition(A, L, U, n);

    // 2) Восстановление A' = L * U и сравнение
    multiplyMatrices(L, U, A_reconstructed, n);
    compareMatrices(A, A_reconstructed, n);

    // Вывод L, U, и A'


    printMatrix(L, n, "Матрица L");
    printMatrix(U, n, "Матрица U");
    printMatrix(A_reconstructed, n, "Восстановленная A = L*U");
    printMatrix(A, n, "Исходная A");


    // 3) и 4) Решение системы Ax = b через LU
    forwardSubstitution(L, b, y, n);
    backwardSubstitution(U, y, x, n);

    printVector(x, n, "Решение системы Ax = b (вектор x)");

    // Освобождение памяти
    delete[] A;
    delete[] b;
    delete[] L;
    delete[] U;
    delete[] A_reconstructed;
    delete[] y;
    delete[] x;

    return 0;
}