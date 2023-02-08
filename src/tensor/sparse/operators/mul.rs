use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::Mul;

pub(crate) fn mul_scalar<T>(slf: T, mut rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    todo!()
}

macro_rules! impl_mul_scalar {
    {$t: ty} => {
        impl Mul<SparseTensor<$t>> for $t {
            type Output = SparseTensor<$t>;

            fn mul(self, rhs: SparseTensor<$t>) -> Self::Output {
                mul_scalar(self, rhs)
            }
        }

        impl Mul<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn mul(self, rhs: SparseTensor<$t>) -> Self::Output {
              mul_scalar(*self, rhs)
          }
        }

        impl Mul<$t> for SparseTensor<$t> {
            type Output = SparseTensor<$t>;

            fn mul(self, rhs: $t) -> Self::Output {
                mul_scalar(rhs, self)
            }
        }

        impl Mul<&$t> for SparseTensor<$t> {
          type Output = SparseTensor<$t>;

          fn mul(self, rhs: &$t) -> Self::Output {
              mul_scalar(*rhs, self)
          }
        }
    };
}

impl_mul_scalar! {f64}
impl_mul_scalar! {c64}
