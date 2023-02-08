use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::Add;

fn add_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    todo!()
}

fn add<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    todo!()
}

macro_rules! impl_add_scalar {
    {$t: ty} => {
        impl Add<SparseTensor<$t>> for $t {
            type Output = SparseTensor<$t>;

            fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
                add_scalar(self, rhs)
            }
        }

        impl Add<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
              add_scalar(*self, rhs)
          }
        }

        impl Add<$t> for SparseTensor<$t> {
            type Output = SparseTensor<$t>;

            fn add(self, rhs: $t) -> Self::Output {
                add_scalar(rhs, self)
            }
        }

        impl Add<&$t> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn add(self, rhs: &$t) -> Self::Output {
              add_scalar(*rhs, self)
          }
        }
    };
}

impl_add_scalar! {f64}
impl_add_scalar! {c64}

macro_rules! impl_add {
  {$t: ty} => {
      impl Add<SparseTensor<$t>> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
            add(self, &rhs)
          }
      }

      impl Add<&SparseTensor<$t>> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn add(self, rhs: &SparseTensor<$t>) -> Self::Output {
            add(self, rhs)
          }
      }

      impl Add<SparseTensor<$t>> for &SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
            add(rhs, self)
          }
      }
  };
}

impl_add! {f64}
impl_add! {c64}
