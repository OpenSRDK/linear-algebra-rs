use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Sub, SubAssign};

fn sub_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 -= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn sub<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
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
      impl Sub<SparseTensor<$t>> for $t {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
              sub_scalar(self, rhs)
          }
      }

      impl Sub<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
              sub_scalar(*self, rhs)
          }
      }
  }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Sub<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        -sub_scalar(rhs, self)
    }
}

impl<T> Sub<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        -sub_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Sub<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: SparseTensor<T>) -> Self::Output {
        sub(self, &rhs)
    }
}

impl<T> Sub<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: &SparseTensor<T>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<T> Sub<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: SparseTensor<T>) -> Self::Output {
        -sub(rhs, self)
    }
}

// SubAssign

impl<T> SubAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn sub_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self - rhs;
    }
}
