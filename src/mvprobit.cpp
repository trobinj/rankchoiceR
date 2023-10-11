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

class binblock
{

private:

  uvec y;
  dvec x;
  dmat u;
  dvec eta;
  dmat sigm;
  int k, q, m, t;

public:

  binblock(uvec y, dvec x) : y(y), x(x)
  {
    k = y.n_elem;
    q = k * (k + 1) / 2;
    
    m = 1;
    t = 5*k;

    u.set_size(m, k);
    for (int j = 0; j < k; ++j) {
      if (y(j) == 1) {
        u(0, j) = rnormpos(0.0, 1.0, 1);
      } else {
        u(0, j) = rnormpos(0.0, 1.0, 0);
      }
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
    double mj, sj;

    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < k; ++j) {
        mj = udist.getm(j, u.t());
        sj = udist.gets(j);
        if (y(j) == 1) {
          u(j) = rnormpos(mj, sj, 1);
        } else {
          u(j) = rnormpos(mj, sj, 0);
        }
      }
    }

    return u;
  }

  dmat getux()
  {
    return mean(u, 0).t() * x.t();
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

  dvec gradient(dmat beta, dmat sigm)
  {
    using namespace arma;

    dmat R = inv(sigm);
    dmat I = eye(k, k);
    dvec z(k);
    dmat betag(size(beta));
    dmat sigmg(size(sigm));

    betag.fill(0.0);
    sigmg.fill(0.0); 

    for (int j = 0; j < t; ++j) {
      z = (u.row(j).t() - beta * x);
      betag = betag + R * z * x.t();
      sigmg = sigmg + 0.5 * (2 * R - (R * I) - 
        2 * R * z * z.t() * R + R * z * z.t() * R * I);
    }

    betag = betag / m;
    sigmg = sigmg / m;

    return join_vert(vectorise(betag), lowertri(sigmg, false));
  }
};

class bindata 
{

private: 

  std::vector<binblock> data;
  int n, p, k, q;
  dmat x;
  dmat beta;
  dmat sigm;
  int ncores;
  dmat xx, xxinv;
  
public:

  bindata(umat y, dmat x, int ncores) : x(x), ncores(ncores)
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
      data.emplace_back(y.row(i).t(), x.row(i).t()); 
    }
  }

  void reduceparameters() 
  {
    dmat D = sqrt(inv(diagmat(sigm)));
  	
  	beta = D * beta;
    sigm = D * sigm * D.t();
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

  void mstep() 
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
    dmat D = sqrt(inv(diagmat(sigm)));
  	
  	beta = D * beta;
    sigm = D * sigm * D.t(); 
  }

  dmat vmat()
  {
    dmat score(n, k * p + k * (k - 1) / 2);
    for (int i = 0; i < n; ++i) {
      score.row(i) = data[i].gradient(beta, sigm).t();
    }
    return inv(score.t() * score);
  }
};

// [[Rcpp::export]]
dmat mvprobit(umat y, dmat x, uvec m, uvec n, int t, int ncores, bool print)
{
  int k = y.n_cols;
	int p = x.n_cols;
  int q = k * (k + 1) / 2; 
  
  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);

  dmat out(n0 + nf, k*p + q);
  
  bindata data(y, x, ncores);
  
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
      Rcpp::Rcout << "iteration: " << i + 1 << "\n";
    	
    	dmat bhat;
    	dmat shat;
    	data.getparameters(bhat, shat);
    	
    	dump<dmat>(bhat, "beta:");
    	dump<dmat>(shat, "sigm:");
    }   

    Rcpp::checkUserInterrupt();
  }

  return out;
}
