extern crate blas;
extern crate lapack;
#[cfg(test)]
extern crate openblas_src;
extern crate rayon;
extern crate thiserror;

pub mod macros;
pub mod matrix;
pub mod number;

pub use crate::{
    matrix::{bd::*, ci::*, di::*, ge::*, kr::*, po::*, pt::*, sp::*, st::*, to::*, tr::*, *},
    number::*,
};
