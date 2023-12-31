# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

mvprobit <- function(y, x, m, n, t, ncores, h, print) {
    .Call(`_rankchoiceR_mvprobit`, y, x, m, n, t, ncores, h, print)
}

rankchoice <- function(y, x, m, n, t, ncores, scale, h, print) {
    .Call(`_rankchoiceR_rankchoice`, y, x, m, n, t, ncores, scale, h, print)
}

rankchoicefull <- function(y, x, r, m, n, t, ncores, scale, h, print) {
    .Call(`_rankchoiceR_rankchoicefull`, y, x, r, m, n, t, ncores, scale, h, print)
}

ranktop <- function(y, x, m, n, t, ncores, scale, h, print) {
    .Call(`_rankchoiceR_ranktop`, y, x, m, n, t, ncores, scale, h, print)
}

tinker <- function(ncores) {
    invisible(.Call(`_rankchoiceR_tinker`, ncores))
}

