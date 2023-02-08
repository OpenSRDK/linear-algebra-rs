use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::Sub;

fn sub_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    todo!()
}

fn sub<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    todo!()
}

macro_rules! impl_sub_scalar {
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

        impl Sub<$t> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: $t) -> Self::Output {
            -sub_scalar(rhs, self)
          }
        }

        impl Sub<&$t> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: &$t) -> Self::Output {
            -sub_scalar(*rhs, self)
          }
        }
    };
}

impl_sub_scalar! {f64}
impl_sub_scalar! {c64}

macro_rules! impl_sub {
  {$t: ty} => {
      impl Sub<SparseTensor<$t>> for SparseTensor<$t> {
        type Output = SparseTensor<$t>;

        fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
          sub(self, &rhs)
        }
      }

      impl Sub<&SparseTensor<$t>> for SparseTensor<$t> {
        type Output = SparseTensor<$t>;

        fn sub(self, rhs: &SparseTensor<$t>) -> Self::Output {
          sub(self, rhs)
        }
      }

      impl Sub<SparseTensor<$t>> for &SparseTensor<$t> {
        type Output = SparseTensor<$t>;

        fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
          -sub(rhs, self)
        }
      }
  };
}

impl_sub! {f64}
impl_sub! {c64}
