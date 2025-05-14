#include <omp.h>
#include <iostream>
#include <cmath>
#include <time.h>
using namespace std;

#define n 2000 // ðàçìåðíîñòü ìàòðèöû A
#define THREADS 1 // êîëè÷åñòâî ïîòîêîâ 

double zapol_matr(int i, int j) {
    if (i == j) return n + 1;
    return 1;
}

int main() {
    double x_k1500[4][1500] = { 0 }, x_k11500[4][1500] = { 0 }, b1500[4][1500] = { 0 }, bm1500[4][1500] = { 0 };

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 1500; ++j) {
            x_k1500[i][j] = 1;
            x_k11500[i][j] = 0;
            b1500[i][j] = i;
        }
    }

    double norm1500 = 1;
    double res[4] = { 0 };
    double vrem[4] = { 0 };
    int numTR;
    double cur;
    for (int i = 0; i < 4; ++i) {
        clock_t start = clock();
        numTR = pow(2, i);
        norm1500 = 1; 
        while (norm1500 > 0.00001) {
            norm1500 = 0;

#pragma omp parallel num_threads(numTR) reduction(+:norm1500)
            {
                int id = omp_get_thread_num();
                int size = omp_get_num_threads();
                int hag = 1500 / size;
                int rem = 1500 % size;
                int start = hag * id + (id < rem ? id : rem);
                int end = start + hag + (id < rem ? 1 : 0);
                if (id == size - 1) end = 1500;

                for (int k = start; k < end; ++k) {
                    double sum1 = 0, sum2 = 0;
                    for (int j = 0; j < k; j++)
                        sum1 += zapol_matr(k, j) * x_k1500[i][j];
                    for (int j = k + 1; j < 1000; j++)
                        sum2 += zapol_matr(k, j) * x_k1500[i][j];
                    x_k11500[i][k] = (b1500[i][k] - sum1 - sum2) / zapol_matr(k, k);
                }

                for (int k = start; k < end; ++k) {
                    norm1500 += (x_k11500[i][k] - x_k1500[i][k]) * (x_k11500[i][k] - x_k1500[i][k]);
                    x_k1500[i][k] = x_k11500[i][k];
                }
            }
        }
        clock_t end = clock();
        double duration = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Total execution time: %.3f seconds\n", duration.count() / 1000.0);
        vrem[i] = duration.count() / 1000.0;

        for (int j = 0; j < 1500; ++j) {
            for (int k = 0; k < 1500; ++k) {
                if (j == k) bm1500[i][j] += x_k1500[i][k] * 1500;
                else bm1500[i][j] += bm1500[i][k] * zapol_matr(k, k);
            }
        }
        res[i] = 0;
        for (int k = 0; k < 1500; ++k) {
            cur = fabs(b1500[i][k] - bm1500[i][k]);
            res[i] = (res[i] <= cur) ? cur : res[i];
        }
    }
    for (int k = 0; k < 4; ++k) {
        cout << "nevazka: " << res[k] << ", " << "time: " << vrem[k] << endl;
    }


    return 0;
}
