extern crate blas;
#[cfg(test)]
extern crate blas_src;
extern crate lapack;
#[cfg(test)]
extern crate lapack_src;
extern crate rand;
extern crate rayon;
extern crate thiserror;

pub mod macros;
pub mod matrix;
pub mod number;
pub mod tensor;

pub use matrix::*;
pub use number::*;
pub use tensor::*;
