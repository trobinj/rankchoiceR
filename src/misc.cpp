#ifndef misc_h
#define misc_h

#include <RcppArmadillo.h>
#include <mvnorm.h>
#include "rankchoiceR_types.h"

dmat cmatrix(int k) 
{
	using namespace arma;
	return join_rows(eye(k - 1, k - 1), -ones(k - 1, 1));
}

rvec rmvnorm(dvec m, dmat s)
{
	return rmvnorm(1, m, s);
}

dmat cov2cor(dmat x) 
{
	dmat d = sqrt(inv(diagmat(x)));
	return d * x * d;
}

dmat vec2lower(dvec x)
{
	int n = (sqrt(8 * x.n_elem + 1) - 1) / 2;
	int t = 0;
	dmat y(n, n);
	for (int j = 0; j < n; ++j) {
		for (int i = j; i < n; ++i) {
			y(i, j) = x(t);
			++t;
		}
	}
	return y;
}

dmat vec2cov(dvec x)
{
	dmat L = vec2lower(x);
	return L * L.t();
}

dvec lowertri(dmat x, bool diag = true) 
{
	if (!diag) x.shed_row(0);
	int n = x.n_rows;
	int m = x.n_cols;
	int d = std::min(n,m) * (std::min(n,m) + 1) / 2;
	if (n > m) {
		d = d + (n - m) * m;
	}
	dvec y(d);
	int t = 0;
	for (int j = 0; j < std::min(n,m); ++j) {
		for (int i = j; i < n; ++i) {
			y(t) = x(i,j);
			++t;
		}
	}
	return y;
}

double rnormpos(double m, double s, bool pos)
{
	double l, a, z, p, u;
	l = pos ? -m/s : m/s;
	a = (l + sqrt(pow(l, 2) + 4.0)) / 2.0;
	
	do {
		z = R::rexp(1.0) / a + l;
		u = R::runif(0.0, 1.0);
		p = exp(-pow(z - a, 2) / 2.0);
	} while (u > p);
	
	return pos ? z * s + m : -z * s + m;
}

double rnormint(double m, double s, double a, double b)
{
	const double sqrt2pi = 2.506628;
	double low = (a - m) / s;
	double upp = (b - m) / s;
	double z, u, p, d;
	
	if (upp < 0) {
		d = pow(upp,2);
	} else if (low > 0) {
		d = pow(low,2);
	} else {
		d = 0.0;
	}
	
	if ((b - a) / d < sqrt2pi) {
		do {
			z = R::runif(low, upp);
			u = R::runif(0.0, 1.0);
			p = exp((d - pow(z,2)) / 2.0);
		} while (u > p);
	} else {
		do {
			z = R::rnorm(0.0, 1.0);
		} while (low > z || z > upp);
	}
	
	return z * s + m;
}

double rtnorm(double m, double s, double a, double b)
{
	bool anan = std::isnan(a);
	bool bnan = std::isnan(b);
	
	if (anan && bnan) {
		return R::rnorm(m, s);
	}
	if (anan) {
		return rnormpos(m - b, s, 0) + b;
	}
	if (bnan) {
		return rnormpos(m - a, s, 1) + a;
	}
	return rnormint(m, s, a, b);
}

class cnorm
{
private:
	
	int n;
	dmat b;
	dvec v;
	dvec m;
	dmat c;
	
public:
	
	cnorm(dmat c) : c(c)
	{
		n = c.n_cols;
		m = arma::zeros(n);
		b.set_size(n, n - 1);
		v.set_size(n);
		setc(c);
	}
	
	cnorm(dvec m, dmat c) : m(m), c(c)
	{
		n = m.n_elem;
		b.set_size(n, n - 1);
		v.set_size(n);
		setc(c);
	}
	
	void setm(dvec x)
	{
		m = x;
	}
	
	void setc(dmat c)
	{
		for (int i = 0; i < n; ++i) {
			dmat c22 = c;
			c22.shed_row(i);
			c22.shed_col(i);
			dmat r22 = inv(c22);
			dmat c12 = c.row(i);
			c12.shed_col(i);
			b.row(i) = c12 * r22;
			v(i) = c(i,i) - as_scalar(b.row(i) * c12.t());
		}
	}
	
  dmat gets()
  {
    return c;
  }

  dvec getm() 
  {
    return m;
  }
	
	double gets(int i) 
	{
		return sqrt(v(i));
	}
	
	double getm(int i, dvec y)
	{
		dvec m2(m);
		dvec y2(y);
		m2.shed_row(i);
		y2.shed_row(i);
		return m(i) + as_scalar(b.row(i) * (y2 - m2));
	}
};

#endif
