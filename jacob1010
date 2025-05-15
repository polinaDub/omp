#include <omp.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <numa.h> 

using namespace std;

#define n 2000

double zapol_matr(int i, int j) {
    if (i == j) return n + 1;
    return 1;
}

int main() {
    // Инициализация NUMA (если доступно)
    if (numa_available() == -1) {
        cerr << "NUMA not available. Running without NUMA optimizations." << endl;
    }

    double x_k10[4][10] = {0}, x_k110[4][10] = {0}, b10[4][10] = {0}, bm10[4][10] = {0};
 
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 10; ++j) {
            x_k10[i][j] = 1;
            x_k110[i][j] = 0;
            b10[i][j] = i;
        }
    }

    double norm10 = 1;
    double res[4] = {0};
    double vrem[4] = {0};
    double cur;
    int numTR;

   
    omp_set_schedule(omp_sched_guided, 0);

    for (int i = 0; i < 4; ++i) {
        clock_t start = clock();
        numTR = pow(2, i);
        norm10 = 1;
        
        
        omp_set_num_threads(numTR);
        #pragma omp parallel
        {
            if (numa_available() != -1) {
                numa_set_localalloc(); // Использовать локальную память NUMA
            }
        }

        while (norm10 > 0.00001) {
            norm10 = 0;

            #pragma omp parallel num_threads(numTR) reduction(+:norm10)
            {
                int id = omp_get_thread_num();
                int size = omp_get_num_threads();
                int hag = 10 / size;
                int rem = 10 % size;
                int start_id = hag * id + (id < rem ? id : rem);
                int end = start_id + hag + (id < rem ? 1 : 0);
                if (id == size - 1) end = 10;

                
                double sum1 = 0, sum2 = 0;
                
                for (int k = start_id; k < end; ++k) {
                    sum1 = 0; sum2 = 0;
                    for (int j = 0; j < k; j++) sum1 += zapol_matr(k, j) * x_k10[i][j];
                    for (int j = k + 1; j < 10; j++) sum2 += zapol_matr(k, j) * x_k10[i][j];
                    x_k110[i][k] = (b10[i][k] - sum1 - sum2) / zapol_matr(k, k);
                }

                for (int k = start_id; k < end; ++k) {
                    norm10 += (x_k110[i][k] - x_k10[i][k]) * (x_k110[i][k] - x_k10[i][k]);
                    x_k10[i][k] = x_k110[i][k];
                }
            }
            norm10 = sqrt(norm10);
        }

        clock_t end = clock();
        double duration = static_cast<double>(end - start) * 1000 / CLOCKS_PER_SEC;
        vrem[i] = duration;
        
        
        #pragma omp parallel for
        for (int j = 0; j < 10; ++j) {
            bm10[i][j] = 0;
            for (int k = 0; k < 10; ++k) {
                if (j == k) bm10[i][j] += x_k10[i][k] * 10;
                else bm10[i][j] += x_k10[i][k] * 1; // Исправлено: было bm10[i][k]
            }
        }

        res[i] = 0;
        for (int k = 0; k < 10; ++k) {
            cur = fabs(b10[i][k] - bm10[i][k]);
            res[i] = max(res[i], cur);
        }
    }

    for (int k = 0; k < 4; ++k) {
        cout << "nevazka: " << res[k] << ", time: " << vrem[k] << " ms" << endl;
    }

    return 0;
}
