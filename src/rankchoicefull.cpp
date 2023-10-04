#include <RcppArmadillo.h>
#include <omp.h>
#include "rankchoiceR_types.h"
#include "misc.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

template <typename type> static void dump(type x)
{
	Rcpp::Rcout << x << "\n";
}

template <typename type> static void dump(type x, std::string y)
{
	Rcpp::Rcout << y << "\n" << x << "\n";
}

static void Rstop(bool x, std::string message)
{
	if (x) Rcpp::stop("message");
}

class rankchoicefullblock
{

private:

  uvec y;
  dvec x;
  dmat u;
  dvec eta;
  dmat sigm;
  int k, q, m, t, r;

  ivec lowindx;
  ivec uppindx;
  uvec yupp;
  uvec ylow;

  const double ninf = arma::datum::nan;
  const double pinf = arma::datum::nan;
  
public:

  rankchoicefullblock(uvec y, int r) : y(y), r(r)
  {
    k = y.n_elem;
    q = k * (k + 1) / 2;
    
    m = 1;
    t = 5*k;

    lowindx.set_size(k);
    uppindx.set_size(k);

		for (int j = 0; j < k; ++j) {
      if (y(j) == 1) {
        uppindx(j) = -1;
        lowindx(j) = as_scalar(find(y == 2));
      } else if (y(j) == k) {
        uppindx(j) = as_scalar(find(y == k - 1));
        lowindx(j) = -1;
      } else {
        uppindx(j) = as_scalar(find(y == y(j) - 1));
        lowindx(j) = as_scalar(find(y == y(j) + 1));
      }
    }

		u.set_size(m, k);
				
    dvec z(k); 
    for (int j = 0; j < k; ++j) {
      if (j < r) {
        z(j) = rtnorm(0.0, 1.0, 0.0, pinf);
      } else {
        z(j) = rtnorm(0.0, 1.0, ninf, 0.0);
      }
    }
    z = sort(z, "descend");
    for (int j = 0; j < k; ++j) {
      u(0, j) = z(y(j) - 1);
    }
  }

  void setsize(int m, int t)
  {
    this->m = m;
    this->t = t;
    u.resize(m, k);
  }

  rvec usamp(rvec u, cnorm& udist)
  {
    double low, upp;
    double mj, sj, yj;

    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < k; ++j) {
        yj = y(j); 
        if (yj == 1) {
          if (yj == r) {
            upp = pinf;
            low = 0.0;
          } else if (yj == r + 1) {
            upp = 0.0;
            low = u(lowindx(j));
          } else {
            upp = pinf;
            low = u(lowindx(j));
          }
        } else if (yj == k) {
          if (yj == r) {
            upp = u(uppindx(j));
            low = 0.0;
          } else if (yj == r + 1) {
            upp = 0.0;
            low = ninf;
          } else {
            upp = u(uppindx(j));
            low = ninf;
          }
        } else {
          if (yj == r) {
            upp = u(uppindx(j));
            low = 0.0;
          } else if (yj == r + 1) {
            upp = 0.0;
            low = u(lowindx(j));
          } else {
            upp = u(uppindx(j));
            low = u(lowindx(j));
          }
        }
        
        mj = udist.getm(j, u.t());
        sj = udist.gets(j);

        u(j) = rtnorm(mj, sj, low, upp);
      }
    }
    
    return u;
  }

  dvec getux()
  {
    return mean(u, 0).t();  
  }

  dmat getuu()
  {
    return (u.t() * u) / m;
  }

  void estep(cnorm& udist)
  {
    for (int j = 0; j < m; ++j) {
      u.row(j) = usamp(u.row(0), udist);
    }
  }
};

class rankchoicefulldata 
{

private: 

  std::vector<rankchoicefullblock> data;
  int n, p, k, q;
  dmat x;
  dmat beta;
  dmat sigm;
  int ncores;
  dmat xx, xxinv;
  double scale;
  
public:

  rankchoicefulldata(umat y, dmat x, uvec r, int ncores, double scale) 
		: x(x), ncores(ncores), scale(scale)
  {
    n = y.n_rows;
    p = x.n_cols;
    k = y.n_cols;
    q = k * (k + 1) / 2;

    beta.zeros(k, p);
    sigm.eye(k, k);

    xx = x.t() * x;
    xxinv = inv(xx);

    data.reserve(n);
    for (int i = 0; i < n; ++i) {
      data.emplace_back(y.row(i).t(), r(i)); 
    }
  }
  
  dvec getparameters()
  {
    dvec theta(k*p + q);
    theta.head(k*p) = vectorise(beta);
    theta.tail(q) = vectorise(lowertri(sigm));
    
    return theta;
  }

	void getparameters(dmat& bhat, dmat& shat)
	{
		bhat = beta;
		shat = sigm;
	}

  void setsize(int m, int t)
  {
    for (auto& obs : data) {
      obs.setsize(m, t);
    }
  }

  void estep() 
  {
  	cnorm udist(sigm);
    dmat eta = beta * x.t();
    
    omp_set_num_threads(ncores);
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < n; ++i) {
    	udist.setm(eta.col(i));
      data[i].estep(udist);
    }
  }

  void mstep() // maybe parallelize this by using cubes or std::vector 
  {
    dmat eux(k, p);
    dmat euu(k, k);

    for (int i = 0; i < n; ++i) {
      eux = eux + data[i].getux();
      euu = euu + data[i].getuu();
    }

    beta = eux * xxinv;
    sigm = (euu - beta * xx * beta.t()) / n;
  }

  void xstep()
  {
  	double d = scale / sigm(0,0);

    sigm = sigm * d;
    beta = beta * sqrt(d);
  }
};

// [[Rcpp::export]]
dmat rankchoicefull(umat y, dmat x, uvec r, uvec m, uvec n, int t, int ncores, double scale, bool print)
{
  int k = y.n_cols;
	int p = x.n_cols;
  int q = k * (k + 1) / 2; 

  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);
  
  dmat out(n0 + nf, k*p + q);
  
  rankchoicefulldata data(y, x, r, ncores, scale);
  
	data.setsize(m0, t);
  
  for (int i = 0; i < n0 + nf; ++i) {

    if ((i + 1) == n0 + 1) {
      data.setsize(mf, t);
    }
  	
    data.estep();
    data.mstep();
		data.xstep();

    out.row(i) = data.getparameters().t();

    if ((i + 1) % 10 == 0 && print) {
      dump<int>(i + 1, "iteration: ");

      dmat bhat, shat;
    	data.getparameters(bhat, shat);

      dump<dmat>(bhat, "beta: \n");
    	dump<dmat>(shat, "sigm: \n");
    }   

    Rcpp::checkUserInterrupt();
  }

  return out;
}