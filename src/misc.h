#ifndef misc_h
#define misc_h

#include <RcppArmadillo.h>
#include <mvnorm.h>
#include "rankchoiceR_types.h"

dmat diagonalize(dmat x, int k = 0);
dmat cmatrix(int k);
rvec rmvnorm(dvec m, dmat s);
dmat cov2cor(dmat x);
dmat vec2lower(dvec x);
dmat vec2cov(dvec x);
dvec lowertri(dmat x, bool diag = true); 
double rnormpos(double m, double s, bool pos);
double rnormint(double m, double s, double a, double b);
double rtnorm(double m, double s, double a, double b);

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
