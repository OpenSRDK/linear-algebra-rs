extern crate blas;
extern crate lapack;
extern crate num_complex;
#[cfg(test)]
extern crate openblas_src;
extern crate rayon;

pub mod circulant_matrix;
pub mod diagonalized;
pub mod lu_decomposed;
pub mod macros;
pub mod matrix;
pub mod number;
pub mod prelude;
pub mod toeplitz_matrix;
pub mod types;
