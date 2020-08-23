extern crate blas;
extern crate lapack;
#[cfg(test)]
extern crate openblas_src;
extern crate rayon;

pub mod macros;
pub mod matrix;
pub mod number;

pub use crate::{
    matrix::{ci::CirculantMatrix, st::SymmetricTridiagonalMatrix, to::ToeplitzMatrix, *},
    number::*,
};
