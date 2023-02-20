use crate::{
    indices_cartesian_product,
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Mul, MulAssign};

fn mul_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 *= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn mul<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    indices_cartesian_product(&lhs.sizes)
        .into_iter()
        .for_each(|k| {
            if !lhs.elems.contains_key(&k) {
                return;
            }
            if !rhs.elems.contains_key(&k) {
                lhs.elems.remove(&k);
                return;
            }
            lhs[&k] *= rhs[&k];
        });

    lhs
}

// Scalar and SparseTensor

macro_rules! impl_div_scalar {
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
  }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Mul<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        mul_scalar(rhs, self)
    }
}

impl<T> Mul<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &T) -> Self::Output {
        mul_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Mul<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        mul(self, &rhs)
    }
}

impl<T> Mul<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        mul(self, rhs)
    }
}

impl<T> Mul<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        mul(rhs, self)
    }
}

// MulAssign

impl<T> MulAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn mul_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::number::Number;

    #[test]
    fn mul_scalar() {
        let mut a = SparseTensor::new(vec![3, 2, 2]);
        a[&[0, 0, 0]] = 2.0;
        a[&[0, 0, 1]] = 4.0;
        a[&[1, 1, 0]] = 2.0;
        a[&[1, 1, 1]] = 4.0;
        a[&[2, 0, 0]] = 2.0;
        a[&[2, 0, 1]] = 4.0;

        let b = 2.0 * a.clone();
        let c = a.clone() * 2.0;
        let d = 2.0 * a;

        // cannot multiply &SparseTensor by Scalar

        // let e = a * 2.0;
        // let f = &a * &2.0;
        // let g = 2.0 * &a;
        // let h = &2.0 * &a;

        assert_eq!(b, c);
        assert_eq!(c, d);
        // assert_eq!(d, e);
        // assert_eq!(e, f);
        // assert_eq!(f, g);
        // assert_eq!(g, h);
    }
}
