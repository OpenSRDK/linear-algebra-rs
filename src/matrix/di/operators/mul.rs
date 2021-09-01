use crate::number::{c64, Number};
use crate::DiagonalMatrix;
use rayon::prelude::*;
use std::ops::Mul;

fn mul_scalar<T>(slf: T, rhs: DiagonalMatrix<T>) -> DiagonalMatrix<T>
where
    T: Number,
{
    let mut rhs = rhs;
    rhs.d
        .par_iter_mut()
        .map(|di| {
            *di *= slf;
        })
        .collect::<Vec<_>>();

    rhs
}

fn mul_di<T>(lhs: DiagonalMatrix<T>, rhs: &DiagonalMatrix<T>) -> DiagonalMatrix<T>
where
    T: Number,
{
    if lhs.n() != rhs.n() {
        panic!("Dimension mismatch.")
    }

    DiagonalMatrix::new(mul_vec(lhs.d, rhs.d()))
}

fn mul_vec<T>(lhs: Vec<T>, rhs: &[T]) -> Vec<T>
where
    T: Number,
{
    if lhs.len() != rhs.len() {
        panic!("Dimension mismatch.")
    }

    let mut lhs = lhs;
    lhs.par_iter_mut()
        .zip(rhs.par_iter())
        .for_each(|(li, &ri)| *li *= ri);

    lhs
}

macro_rules! impl_mul_scalar {
    {$t: ty} => {
        impl Mul<DiagonalMatrix<$t>> for $t {
            type Output = DiagonalMatrix<$t>;

            fn mul(self, rhs: DiagonalMatrix<$t>) -> Self::Output {
                mul_scalar(self, rhs)
            }
        }

        impl Mul<$t> for DiagonalMatrix<$t> {
            type Output = DiagonalMatrix<$t>;

            fn mul(self, rhs: $t) -> Self::Output {
                mul_scalar(rhs, self)
            }
        }
    };
}

impl_mul_scalar! {f64}
impl_mul_scalar! {c64}

macro_rules! impl_mul_di {
  {$t: ty, $e: expr} => {
      impl Mul<DiagonalMatrix<$t>> for DiagonalMatrix<$t> {
          type Output = DiagonalMatrix<$t>;

          fn mul(self, rhs: DiagonalMatrix<$t>) -> Self::Output {
              $e(self, &rhs)
          }
      }

      impl Mul<&DiagonalMatrix<$t>> for DiagonalMatrix<$t> {
          type Output = DiagonalMatrix<$t>;

          fn mul(self, rhs: &DiagonalMatrix<$t>) -> Self::Output {
              $e(self, rhs)
          }
      }

      impl Mul<DiagonalMatrix<$t>> for &DiagonalMatrix<$t> {
        type Output = DiagonalMatrix<$t>;

        fn mul(self, rhs: DiagonalMatrix<$t>) -> Self::Output {
            $e(rhs, self)
        }
      }
  };
}

impl_mul_di! {f64, mul_di}
impl_mul_di! {c64, mul_di}

macro_rules! impl_mul_vec {
  {$t: ty, $e: expr} => {
      impl Mul<Vec<$t>> for DiagonalMatrix<$t> {
          type Output = Vec<$t>;

          fn mul(self, rhs: Vec<$t>) -> Self::Output {
              $e(self.d, &rhs)
          }
      }

      impl Mul<&Vec<$t>> for DiagonalMatrix<$t> {
        type Output = Vec<$t>;

        fn mul(self, rhs: &Vec<$t>) -> Self::Output {
            $e(self.d, rhs)
        }
    }
      impl Mul<Vec<$t>> for &DiagonalMatrix<$t> {
          type Output = Vec<$t>;

          fn mul(self, rhs: Vec<$t>) -> Self::Output {
              $e(rhs, self.d())
          }
      }
  };
}

impl_mul_vec! {f64, mul_vec}
impl_mul_vec! {c64, mul_vec}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) * mat!(
            5.0, 6.0;
            7.0, 8.0
        );
        assert_eq!(a[(0, 0)], 19.0);
    }
}
