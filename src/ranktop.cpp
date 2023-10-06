#include <RcppArmadillo.h>
#include <roptim.h>
#include <omp.h>
#include "rankchoiceR_types.h"
#include "misc.h"

using namespace roptim;

// [[Rcpp::depends(RcppArmadillo, RcppDist, roptim)]]
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
	if (x) Rcpp::Rcout << message << "\n";
}

class parameterdata : public Functor 
{
private:

  int k, p;
  dmat D;
  dmat beta;
  dmat sigm;
  double scale;

public:

  parameterdata(dmat beta, dmat sigm, double scale) 
		: beta(beta), sigm(sigm), scale(scale)
  {
    k = beta.n_rows;
    p = beta.n_cols;
    D = cmatrix(k);
  }

  void setparameters(dvec theta, dmat& bnew, dmat& snew)
  {
    bnew.fill(0.0);
    bnew.head_rows(k - 1) = theta.head(p * (k - 1));

    dvec lamb((k - 1) * k / 2);
    lamb(0) = sqrt(scale);
    lamb.tail((k - 1) * k / 2 - 1) = theta.tail((k - 1) * k / 2 - 1);
    snew.fill(0.0);
    snew.submat(0, 0, k - 2, k - 2) = vec2cov(lamb);
    snew(k - 1, k - 1) = 1.0;
  }
	
	double getscalefactor(double scale) 
	{
		dmat tmp = D * sigm * D.t();
		return 1 / tmp(0,0) * (scale + 1); 
	}

  double operator()(const dvec& theta) override
  {
    dmat bnew(k, p);
    dmat snew(k, k);
    setparameters(theta, bnew, snew);
    
		double d = getscalefactor(scale);
    dmat stmp = sigm * d;
    dmat btmp = beta * sqrt(d);

    double lbeta = accu(square(D * (bnew - btmp)));
    double lsigm = accu(square(lowertri(D * (stmp - snew) * D.t())));

    return lbeta + lsigm;
  }
};

class ranktopblock
{

private:

  uvec y;
  dvec x;
  dmat u;
  dvec eta;
  dmat sigm;
  int k, p, q, m, t, r;

  ivec lowindx;
  ivec uppindx;
  uvec yupp;
  uvec ylow;

public:

  ranktopblock(uvec y, dvec x) : y(y), x(x)
  {
    k = y.n_elem;
    p = x.n_elem;
    q = k * (k + 1) / 2;
    r = max(y);
    
    m = 1;
    t = 5*k;

    yupp = find(y != 0);
    ylow = find(y == 0); 
    lowindx.set_size(k);
    uppindx.set_size(k);

    for (int j = 0; j < k; ++j) {
      if (y(j) == 1) {
        if (r == 1) {
          uppindx(j) = -1;
          lowindx(j) = -1;
        } else {
          uppindx(j) = -1;
          lowindx(j) = as_scalar(find(y == 2));
        }
      } else if (y(j) == k) {
        uppindx(j) = as_scalar(find(y == k - 1));
        lowindx(j) = -1;
      } else if (y(j) == 0) {
        uppindx(j) = -1;
        lowindx(j) = -1;
      } else if (y(j) == r) {
        uppindx(j) = as_scalar(find(y == y(j) - 1)); 
        lowindx(j) = -1;
      } else {
        uppindx(j) = as_scalar(find(y == y(j) - 1));
        lowindx(j) = as_scalar(find(y == y(j) + 1));
      }
    }

    u.set_size(m, k);
    if (r == 0) {
      for (int j = 0; j < k; ++j) {
        u(0, j) = R::rnorm(0.0, 1.0);
      }
    } else {
      dvec z(r + 1, arma::fill::randn);
      z = sort(z, "descend");
      for (int j = 0; j < k; ++j) {
        if (y(j) > 0) {
          u(0, j) = z(y(j) - 1);
        } else {
          u(0, j) = z(r);
        }
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
    double low, upp;
    double mj, sj, yj;
    const double ninf = arma::datum::nan;
    const double pinf = arma::datum::nan;

    if (r == 0) {
      u = rmvnorm(udist.getm(), udist.gets());
      return u;
    } 

    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < k; ++j) {
        yj = y(j);
        if (yj == 1) {
          if (r == 1) {
            low = u.elem(ylow).max();
            upp = pinf;
          } else {
            low = u(lowindx(j));
            upp = pinf;
          }
        } else if (yj == k) {
          low = ninf;
          upp = u(uppindx(j));
        } else if (yj == 0) {
          low = ninf;
          upp = u.elem(yupp).min();
        } else if (yj == r) {
          low = u.elem(ylow).max();
          upp = u(uppindx(j));
        } else {
          low = u(lowindx(j));
          upp = u(uppindx(j));
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

    for (int j = 0; j < t; ++j) {
      z = (u.row(j).t() - beta * x);
      betag = betag + R * z * x.t();
      sigmg = sigmg + 0.5 * (2 * R - (R * I) - 
        2 * R * z * z.t() * R + R * z * z.t() * R * I);
    }

    betag = betag / m;
    sigmg = sigmg / m;

    betag.shed_row(k - 1);
    sigmg.shed_row(k - 1);
    sigmg.shed_col(k - 1);

    sigmv = lowertri(sigmg);
    sigmv = sigmv.tail(k * (k - 1) / 2 - 1);

    return join_vert(vectorise(betag), sigmv);
  }
};

class ranktopdata 
{

private: 

  std::vector<ranktopblock> data;
  int n, p, k, q;
  dmat x;
  dmat beta;
  dmat sigm;
  int ncores;
  dmat xx, xxinv;
  double scale;
  
public:

  ranktopdata(umat y, dmat x, int ncores, double scale) 
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
      data.emplace_back(y.row(i).t(), x.row(i).t()); 
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
    dvec theta(p * (k - 1) + (k - 1) * k / 2 - 1);
    theta.head(p * (k - 1)).fill(0.0);
    theta.tail((k - 1) * k / 2 - 1) = lowertri(arma::eye(k - 1, k - 1)).tail((k - 1) * k / 2 - 1);

    parameterdata data(beta, sigm, scale);
    Roptim<parameterdata> opt("BFGS");
    opt.control.reltol = 1e-08;
    opt.control.abstol = 1e-08;
    opt.minimize(data, theta);

    Rstop(opt.value() > 1e-5, "parameter reduction error");

    data.setparameters(opt.par(), beta, sigm);
  }

  dmat vmat()
  {
    dmat score(n, p * (k - 1) + (k - 1) * k / 2 - 1);
    for (int i = 0; i < n; ++i) {
      score.row(i) = data[i].gradient(beta, sigm).t();
    }
    return inv(score.t() * score);
  }
};

// [[Rcpp::export]]
dmat ranktop(umat y, dmat x, uvec m, uvec n, int t, int ncores, double scale, bool print)
{
  int k = y.n_cols;
	int p = x.n_cols;
  int q = k * (k + 1) / 2; 
  
  int m0 = m(0);
  int mf = m(1);
  int n0 = n(0);
  int nf = n(1);

  dmat out(n0 + nf, k*p + q);
  
  ranktopdata data(y, x, ncores, scale);
  
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
