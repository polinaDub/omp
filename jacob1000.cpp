#include <omp.h>
#include <iostream>
#include <cmath>
#include <сtime>
#include<stdlib.h>
using namespace std;


#define n 2000 

double zapol_matr(int i, int j) {
    if (i == j) return n + 1;
    return 1;
}

int main() {
    double x_k1000[4][1000] = { 0 }, x_k11000[4][1000] = { 0 }, b1000[4][1000] = { 0 }, bm1000[4][1000] = { 0 };

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 1000; ++j) {
            x_k1000[i][j] = 1;
            x_k11000[i][j] = 0;
            b1000[i][j] = i;
        }
    }

    double norm1000 = 1;
    double res[4] = {0};  
    double vrem[4] = {0}; 
    int numTR;
    double cur;
    for (int i = 0; i < 4; ++i) {
        clock_t start = clock();
        numTR = pow(2, i);
        norm1000 = 1; 
        while (norm1000 > 0.00001) {
            norm1000 = 0; 

#pragma omp parallel num_threads(numTR) reduction(+:norm1000)
            {
                int id = omp_get_thread_num();
                int size = omp_get_num_threads();
                int hag = 1000 / size;
                int rem = 1000 % size;
                int start_id = hag * id + (id < rem ? id : rem);
                int end = start_id + hag + (id < rem ? 1 : 0);
                if (id == size - 1) end = 1000;

                for (int k = start_id; k < end; ++k) {
                    double sum1 = 0, sum2 = 0;
                    for (int j = 0; j < k; j++)
                        sum1 += zapol_matr(k, j) * x_k1000[i][j];
                    for (int j = k + 1; j < 1000; j++) 
                        sum2 += zapol_matr(k, j) * x_k1000[i][j];
                    x_k11000[i][k] = (b1000[i][k] - sum1 - sum2) / zapol_matr(k, k);
                }

                for (int k = start_id; k < end; ++k) {
                    norm1000 += (x_k11000[i][k] - x_k1000[i][k]) * (x_k11000[i][k] - x_k1000[i][k]);
                    x_k1000[i][k] = x_k11000[i][k];
                }
            }
        }
        clock_t end = clock();
        double duration = static_cas<double>(end - start)* 1000 / CLOCKS_PER_SEC;
        vrem[i] = duration;
        
        for (int j = 0; j < 1000; ++j) {
            for (int k = 0; k < 1000; ++k) {
                if (j == k) bm1000[i][j] += x_k1000[i][k] * 1000;
                else bm1000[i][j] += bm1000[i][k] * zapol_matr(k, k);
            }
        }
        res[i] = 0;
        for (int k = 0; k < 1000; ++k) {
            cur = fabs(b1000[i][k] - bm1000[i][k]);
            res[i] = (res[i] <= cur) ? cur : res[i];
        }
    }
    for (int k = 0; k < 4; ++k) {
        cout << "nevazka: " << res[k] << ", " << "time: " << vrem[k] << endl;
    }
   

    return 0;
}
