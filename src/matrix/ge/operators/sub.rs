use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::{Sub, SubAssign};

fn sub_scalar<T>(lhs: T, rhs: Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r -= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn sub<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
            *l -= r;
        })
        .collect::<Vec<_>>();

    lhs
}

// Scalar and Matrix

macro_rules! impl_div_scalar {
    {$t: ty} => {
        impl Sub<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn sub(self, rhs: Matrix<$t>) -> Self::Output {
                sub_scalar(self, rhs)
            }
        }

        impl Sub<Matrix<$t>> for &$t {
            type Output = Matrix<$t>;

            fn sub(self, rhs: Matrix<$t>) -> Self::Output {
                sub_scalar(*self, rhs)
            }
        }
    }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// Matrix and Scalar

impl<T> Sub<T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: T) -> Self::Output {
        -sub_scalar(rhs, self)
    }
}

impl<T> Sub<&T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        -sub_scalar(*rhs, self)
    }
}

// Matrix and Matrix

impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        sub(self, &rhs)
    }
}

impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        -sub(rhs, self)
    }
}

// SubAssign

impl<T> SubAssign<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        *self = self as &Self - rhs;
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            5.0, 6.0;
            7.0, 8.0
        ) - mat!(
            1.0, 3.0;
            5.0, 7.0
        );
        assert_eq!(a[(0, 0)], 4.0);
        assert_eq!(a[(0, 1)], 3.0);
        assert_eq!(a[(1, 0)], 2.0);
        assert_eq!(a[(1, 1)], 1.0);
    }
}
