#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

inline int idx(int i, int j, int M) { return i * M + j; }

// Одно решение Гаусса–Зейделя для заданного N и заданного числа потоков
double solve_gauss_seidel(int N, int threads, double eps = 1e-4) {
    const double h = 1.0 / (N + 1.0);
    const double h2 = h * h;
    const int M = N + 2;

    std::vector<double> u((N + 2) * (N + 2), 0.0);

    omp_set_num_threads(threads);

    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    double start = omp_get_wtime();
    double dmax;
    int iters = 0;

    do {
        dmax = 0.0;

        for (int color = 0; color < 2; color++) {
#pragma omp parallel
            {
                double local_dm = 0.0;

#pragma omp for schedule(static)
                for (int i = 1; i <= N; i++) {
                    for (int j = 1; j <= N; j++) {
                        if (((i + j) & 1) != color) continue;

                        int k = idx(i, j, M);
                        double old = u[k];
                        double newv = 0.25 * (
                            u[idx(i - 1, j, M)] +
                            u[idx(i + 1, j, M)] +
                            u[idx(i, j - 1, M)] +
                            u[idx(i, j + 1, M)] -
                            h2 * 1.0
                            );

                        u[k] = newv;
                        double d = fabs(newv - old);
                        if (d > local_dm) local_dm = d;
                    }
                }

                omp_set_lock(&dmax_lock);
                if (local_dm > dmax) dmax = local_dm;
                omp_unset_lock(&dmax_lock);
            }
        }

        iters++;

    } while (dmax > eps);

    double end = omp_get_wtime();
    omp_destroy_lock(&dmax_lock);

    return end - start;  // вернуть время
}

int main() {
    std::vector<int> Ns = { 10, 100, 1000, 2000, 3000 };
    std::vector<int> threads = { 1, 2, 4, 8 };

    std::cout << "N\tThreads\tTime(s)\tSpeedup\n";

    for (int N : Ns) {
        double T1 = solve_gauss_seidel(N, 1);

        for (int t : threads) {
            double Tp = (t == 1) ? T1 : solve_gauss_seidel(N, t);
            double S = T1 / Tp;

            std::cout << N << "\t" << t << "\t" << Tp << "\t" << S << "\n";
        }
        std::cout << "-----------------------------\n";
    }

    return 0;
}