// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "rankchoiceR_types.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// mvprobit
dmat mvprobit(umat y, dmat x, uvec m, uvec n, int t, int ncores, bool print);
RcppExport SEXP _rankchoiceR_mvprobit(SEXP ySEXP, SEXP xSEXP, SEXP mSEXP, SEXP nSEXP, SEXP tSEXP, SEXP ncoresSEXP, SEXP printSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< umat >::type y(ySEXP);
    Rcpp::traits::input_parameter< dmat >::type x(xSEXP);
    Rcpp::traits::input_parameter< uvec >::type m(mSEXP);
    Rcpp::traits::input_parameter< uvec >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< bool >::type print(printSEXP);
    rcpp_result_gen = Rcpp::wrap(mvprobit(y, x, m, n, t, ncores, print));
    return rcpp_result_gen;
END_RCPP
}
// rankchoice
dmat rankchoice(umat y, dmat x, uvec m, uvec n, int t, int ncores, double scale, bool print);
RcppExport SEXP _rankchoiceR_rankchoice(SEXP ySEXP, SEXP xSEXP, SEXP mSEXP, SEXP nSEXP, SEXP tSEXP, SEXP ncoresSEXP, SEXP scaleSEXP, SEXP printSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< umat >::type y(ySEXP);
    Rcpp::traits::input_parameter< dmat >::type x(xSEXP);
    Rcpp::traits::input_parameter< uvec >::type m(mSEXP);
    Rcpp::traits::input_parameter< uvec >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< bool >::type print(printSEXP);
    rcpp_result_gen = Rcpp::wrap(rankchoice(y, x, m, n, t, ncores, scale, print));
    return rcpp_result_gen;
END_RCPP
}
// ranktop
dmat ranktop(umat y, dmat x, uvec m, uvec n, int t, int ncores, double scale, bool print);
RcppExport SEXP _rankchoiceR_ranktop(SEXP ySEXP, SEXP xSEXP, SEXP mSEXP, SEXP nSEXP, SEXP tSEXP, SEXP ncoresSEXP, SEXP scaleSEXP, SEXP printSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< umat >::type y(ySEXP);
    Rcpp::traits::input_parameter< dmat >::type x(xSEXP);
    Rcpp::traits::input_parameter< uvec >::type m(mSEXP);
    Rcpp::traits::input_parameter< uvec >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< bool >::type print(printSEXP);
    rcpp_result_gen = Rcpp::wrap(ranktop(y, x, m, n, t, ncores, scale, print));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rankchoiceR_mvprobit", (DL_FUNC) &_rankchoiceR_mvprobit, 7},
    {"_rankchoiceR_rankchoice", (DL_FUNC) &_rankchoiceR_rankchoice, 8},
    {"_rankchoiceR_ranktop", (DL_FUNC) &_rankchoiceR_ranktop, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_rankchoiceR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
