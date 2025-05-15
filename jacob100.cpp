#include <omp.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include<stdlib.>
using namespace std;

#define n 2000 
double zapol_matr(int i, int j) {
    if (i == j) return n + 1;
    return 1;
}

int main() {

    double x_k10[4][10] = { 0 }, x_k110[4][10] = { 0 }, b10[4][10] = { 0 }, bm10[4][10] = { 0 };
    double x_k100[4][100] = { 0 }, x_k1100[4][100] = { 0 }, b100[4][100] = { 0 }, bm100[4][100] = { 0 };
    double x_k1000[4][1000] = { 0 }, x_k11000[4][1000] = { 0 }, b1000[4][1000] = { 0 }, bm1000[4][1000] = { 0 };
    double x_k1500[4][1500] = { 0 }, x_k11500[4][1500] = { 0 }, b1500[4][1500] = { 0 }, bm1500[4][1500] = { 0 };
    double x_k2000[4][2000] = { 0 }, x_k12000[4][2000] = { 0 }, b2000[4][2000] = { 0 }, bm2000[4][2000] = { 0 };

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 100; ++j) {
            x_k100[i][j] = 1;
            x_k1100[i][j] = 0;
            b100[i][j] = i;
        }
    }

    double  norm100 = 1;
    double res[4] = { 0 };
    double vrem[4] = { 0 };
    double cur;
    int numTR;

    for (int i = 0; i < 4; ++i) {
        clock_t start = clock();
        numTR = pow(2, i);
        norm100 = 1;
        while (norm100 > 0.00001) {
            norm100 = 0;

#pragma omp parallel num_threads(numTR) reduction(+:norm100)
            {
                int id = omp_get_thread_num();
                int size = omp_get_num_threads();
                int hag = 100 / size;
                int rem = 100 % size;
                int start_id = hag * id + (id < rem ? id : rem);
                int end = start_id + hag + (id < rem ? 1 : 0);
                if (id == size - 1) end = 100;

                for (int k = start_id; k < end; ++k) {
                    double sum1 = 0, sum2 = 0;
                    for (int j = 0; j < k; j++) sum1 += zapol_matr(k, j) * x_k100[i][j];
                    for (int j = k + 1; j < 100; j++) sum2 += zapol_matr(k, j) * x_k100[i][j];
                    x_k1100[i][k] = (b100[i][k] - sum1 - sum2) / zapol_matr(k, k);
                }

                for (int k = start_id; k < end; ++k) {
                    norm100 += (x_k1100[i][k] - x_k10[i][k]) * (x_k1100[i][k] - x_k100[i][k]);
                    x_k100[i][k] = x_k1100[i][k];
                }
            }
        }
        clock_t end = clock();
        double duration = static_cas<(double>(end - start)* 1000 / CLOCKS_PER_SEC;
        vrem[i] = duration;
        
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 100; ++k) {
                if (j == k) bm100[i][j] += x_k100[i][k] * 100;
                else bm100[i][j] += bm100[i][k] * zapol_matr(k, k);
            }
        }
        res[i] = 0;
        for (int k = 0; k < 100; ++k) {
            cur = fabs(b100[i][k] - bm100[i][k]);
            res[i] = (res[i] <= cur) ? cur : res[i];
        }
    }
    for (int k = 0; k < 4; ++k) {
        cout << "nevazka: " << res[k] << ", " << "time: " << vrem[k] << endl;
    }
    return 0;
}
