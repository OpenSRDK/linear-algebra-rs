use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::Add;

fn add_scalar<T>(lhs: T, rhs: Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r += lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn add<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    lhs.elems
        .par_iter_mut()
        .zip(rhs.elems.par_iter())
        .map(|(l, &r)| {
            *l += r;
        })
        .collect::<Vec<_>>();

    lhs
}

macro_rules! impl_add_scalar {
    {$t: ty} => {
        impl Add<Matrix<$t>> for $t {
          type Output = Matrix<$t>;

          fn add(self, rhs: Matrix<$t>) -> Self::Output {
            add_scalar(self, rhs)
          }
        }

        impl Add<Matrix<$t>> for &$t {
          type Output = Matrix<$t>;

          fn add(self, rhs: Matrix<$t>) -> Self::Output {
            add_scalar(*self, rhs)
          }
        }

        impl Add<$t> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn add(self, rhs: $t) -> Self::Output {
            add_scalar(rhs, self)
          }
        }

        impl Add<&$t> for Matrix<$t> {
          type Output = Matrix<$t>;

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
      impl Add<Matrix<$t>> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn add(self, rhs: Matrix<$t>) -> Self::Output {
            add(self, &rhs)
          }
      }

      impl Add<&Matrix<$t>> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn add(self, rhs: &Matrix<$t>) -> Self::Output {
            add(self, rhs)
          }
      }

      impl Add<Matrix<$t>> for &Matrix<$t> {
          type Output = Matrix<$t>;

          fn add(self, rhs: Matrix<$t>) -> Self::Output {
            add(rhs, self)
          }
      }
  };
}

impl_add! {f64}
impl_add! {c64}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) + mat!(
            5.0, 6.0;
            7.0, 8.0
        );
        assert_eq!(a[(0, 0)], 6.0);
        assert_eq!(a[(0, 1)], 8.0);
        assert_eq!(a[(1, 0)], 10.0);
        assert_eq!(a[(1, 1)], 12.0);
    }
}
