#include <iostream>
#include <math.h>
#include <stdio.h>
#include <ctime>

using namespace std;

#define  n  100
#define  r  10

static double A[n][n], U[n][n], L[n][n];

void zpoln_matr() {
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i][j] = (i == j) ? n : 1;  // Íà äèàãîíàëè n, îñòàëüíîå 1
			U[i][j] = 0; //íóëåâàÿ ìàòðèöà
			L[i][j] = (i == j) ? 1 : 0; // åäèíè÷íàÿ ìàòðèöà
		}
	}
}

static void razlogenie_lu(int i) {// LU-ðàçëîæåíèå äëÿ áëîêà ìàòðèöû ðàçìåðîì r, íà÷èíàÿ (i,i)
	int end = (i + r < n) ? i + r : n;

	double sum1 = 0; //âû÷èñëåíå çíà÷åíèé âåðøíåòðåóãîëüíîé ìàòðèöû ðåàëèçàöèÿ ìåòîäà ãàóññà äëÿ äëî÷íîãî ðàçëîæåíèÿ
	for (int l = i; l < end; ++l) {
		for (int k = i; k <= l; ++k) {
			sum1 = 0;
			for (int t = i; t < k; ++t) // Âû÷èñëÿåì U[k][j] äëÿ j >= k
				sum1 += L[k][t] * U[t][l];
			U[k][l] = A[k][l] - sum1;
		}
		for (int k = i + 1; k < end; ++k) { //âû÷èñëåíå çíà÷åíèé íèæíåòðåóãîëüíîé ìàòðèöû
			sum1 = 0;
			for (int t = i; t < l; ++t) // Âû÷èñëÿåì L[i][k] äëÿ i > k
				sum1 += L[k][t] * U[t][l];
			L[k][l] = (A[k][l] - sum1) / U[l][l];
		}
	}
}
static void rewie_U(int i, int j) { // îáíîâëåíèå ñïðàâà îò áëîêà
	double s1 = 0;
	for (int l = j; l < n; ++l) {
		for (int k = i; k < i + r; ++k) {
			s1 = 0;
			for (int s = i; s < k; ++s)
				s1 += L[k][s] * U[s][l];
			U[k][l] = A[k][l] - s1;
		}
	}
}
static void rewie_L(int i, int j) { // îáíîâëåíèå ïîä òåêóùèì áëîêîì
	double s1 = 0;
	for (int k = i; k < n; ++k) {
		for (int l = j + r - 1; l >= j; --l) {
			s1 = 0;
			for (int s = l + 1; s < j + r; ++s)
				s1 += L[k][s] * U[s][l];
			L[k][l] = (A[k][l] - s1) / U[l][l];
		}
	}
}
static void reductia(int i) {
	double s1 = 0;
	for (int k = i; k < n; ++k) {
		for (int l = i; l < n; ++l) {
			s1 = 0;
			for (int s = i - r; s < i; ++s)
				s1 += L[k][s] * U[s][l];
			A[k][l] -= s1;
		}
	}
}

static void lu(int i) {
	if (n - i <= r) // åñëè òåêóùèé áëîê ìåíüøå r
		razlogenie_lu(i);
	if (i >= n) return; // áëîê äåëèòñÿ è äàëüøå âûïîëíÿåòñÿ àëãîðèòì. 

#pragma omp task // ñîçäàíèå çàäà÷ 
	razlogenie_lu(i);
#pragma omp taskwait
#pragma omp task
	rewie_U(i, i + r);
#pragma omp task
	rewie_L(i + r, i);
#pragma omp taskwait

	reductia(i + r);
	lu(i + r);
}

int main() {
	for (int num = 0; num < 4; ++num) {
		zpoln_matr();

		int num_tread = pow(2, num);
		clock_t start_time = clock(); // íà÷àëî âðåìåíè

#pragma omp parallel num_threads(num_tread)
#pragma omp single nowait
		lu(0);

		clock_t end_time = clock(); 

		cout << "time: ";
		printf("%.6lf\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);
	}

	return 0;
}
