use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Div, DivAssign};

fn div_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 /= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn div<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    todo!();

    lhs
}

// Scalar and SparseTensor

macro_rules! impl_div_scalar {
  {$t: ty} => {
      impl Div<SparseTensor<$t>> for $t {
          type Output = SparseTensor<$t>;

          fn div(self, rhs: SparseTensor<$t>) -> Self::Output {
              div_scalar(self, rhs)
          }
      }

      impl Div<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn div(self, rhs: SparseTensor<$t>) -> Self::Output {
              div_scalar(*self, rhs)
          }
      }
  }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Div<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        div_scalar(rhs, self)
    }
}

impl<T> Div<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: &T) -> Self::Output {
        div_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Div<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: SparseTensor<T>) -> Self::Output {
        div(self, &rhs)
    }
}

impl<T> Div<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: &SparseTensor<T>) -> Self::Output {
        div(self, rhs)
    }
}

impl<T> Div<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: SparseTensor<T>) -> Self::Output {
        div(rhs, self)
    }
}

// DivAssign

impl<T> DivAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn div_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self / rhs;
    }
}
