extern crate blas;
#[cfg(test)]
extern crate blas_src;
extern crate lapack;
#[cfg(test)]
extern crate lapack_src;
extern crate rayon;
extern crate thiserror;

pub use crate::{
    matrix::{
        bd::*,
        ci::*,
        di::*,
        ge::{
            or_un::*,
            sy_he::{po::*, *},
            tr::*,
            *,
        },
        gt::*,
        kr::*,
        sp_hp::{pp::*, *},
        ss::*,
        st::{pt::*, *},
        to::*,
        *,
    },
    number::*,
};

pub mod macros;
pub mod matrix;
pub mod number;
