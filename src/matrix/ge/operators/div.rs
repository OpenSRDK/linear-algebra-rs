use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::Div;

pub(crate) fn div_scalar<T>(mut slf: Matrix<T>, rhs: T) -> Matrix<T>
where
    T: Number,
{
    slf.elems.par_iter_mut().for_each(|l| {
        *l /= rhs;
    });

    slf
}

macro_rules! impl_div_scalar {
  {$t: ty} => {
      impl Div<$t> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn div(self, rhs: $t) -> Self::Output {
              div_scalar(self, rhs)
          }
      }

      impl Div<&$t> for Matrix<$t> {
        type Output = Matrix<$t>;

        fn div(self, rhs: &$t) -> Self::Output {
            div_scalar(self, *rhs)
        }
      }
  };
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) / 2.0;
        assert_eq!(a[(0, 0)], 0.5);
    }
}
