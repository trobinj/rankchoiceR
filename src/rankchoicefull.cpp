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

  rankchoicefullblock(uvec y, dvec x, int r) : y(y), x(x), r(r)
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
    double mj, sj;
    int yj;

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
    dvec sigmv;

    betag.fill(0.0);
    sigmg.fill(0.0);

    for (int j = 0; j < m; ++j) {
      z = (u.row(j).t() - beta * x);
      betag = betag + R * z * x.t();
      sigmg = sigmg + 0.5 * (2 * R - (R * I) - 
        2 * R * z * z.t() * R + R * z * z.t() * R * I);
    }

    betag = betag / m;
    sigmg = sigmg / m;

    sigmv = lowertri(sigmg);
    sigmv = sigmv.tail(k * (k + 1) / 2 - 1);

    return join_vert(vectorise(betag), sigmv);
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
      data.emplace_back(y.row(i).t(), x.row(i).t(), r(i)); 
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

  void setparameters(dvec theta)
  {
    dmat temp = theta.head(k*p);
    beta = temp.reshape(size(beta));
    sigm = vec2lower(theta.tail(q), true);
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

  dmat vmat()
  {
    dmat score(n, k * p + k * (k + 1) / 2 - 1);
    for (int i = 0; i < n; ++i) {
      score.row(i) = data[i].gradient(beta, sigm).t();
    }
    return inv(score.t() * score);
  }
};

// [[Rcpp::export]]
Rcpp::List rankchoicefull(umat y, dmat x, uvec r, uvec m, uvec n, int t, int ncores, double scale, int h, bool print)
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

  dvec theta = mean(out.tail_rows(nf/2), 0).t();

  dmat vcov;
  if (h) {
    data.setparameters(theta); 
    data.setsize(h, t);
    data.estep();
    vcov = data.vmat();
  } else {
    vcov.fill(arma::datum::nan);
  }

  return Rcpp::List::create(
    Rcpp::Named("out") = Rcpp::wrap(out),
    Rcpp::Named("theta") = Rcpp::wrap(theta),
    Rcpp::Named("vcov") = Rcpp::wrap(vcov)
  );
}
