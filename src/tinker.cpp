#include <RcppArmadillo.h>
#include <omp.h>
#include <iostream>

using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
void tinker(int ncores)
{
	int id, numthrds;
	
	omp_set_num_threads(ncores);
	#pragma omp parallel 
	{
	id = omp_get_thread_num();
	numthrds = omp_get_num_threads();
	cout << "Hello from thread " << id << " of " << numthrds << endl;
	}
}
