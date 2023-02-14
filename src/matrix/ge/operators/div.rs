use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::{Div, DivAssign};

fn div_scalar<T>(lhs: T, rhs: Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r /= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn div<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
            *l /= r;
        })
        .collect::<Vec<_>>();

    lhs
}

// Scalar and Matrix

macro_rules! impl_div_scalar {
    {$t: ty} => {
        impl Div<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn div(self, rhs: Matrix<$t>) -> Self::Output {
                div_scalar(self, rhs)
            }
        }

        impl Div<Matrix<$t>> for &$t {
            type Output = Matrix<$t>;

            fn div(self, rhs: Matrix<$t>) -> Self::Output {
                div_scalar(*self, rhs)
            }
        }
    }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// Matrix and Scalar

impl<T> Div<T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        div_scalar(rhs, self)
    }
}

impl<T> Div<&T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn div(self, rhs: &T) -> Self::Output {
        div_scalar(*rhs, self)
    }
}

// Matrix and Matrix

impl<T> Div<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn div(self, rhs: Matrix<T>) -> Self::Output {
        div(self, &rhs)
    }
}

impl<T> Div<&Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn div(self, rhs: &Matrix<T>) -> Self::Output {
        div(self, rhs)
    }
}

impl<T> Div<Matrix<T>> for &Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn div(self, rhs: Matrix<T>) -> Self::Output {
        div(rhs, self)
    }
}

// DivAssign

impl<T> DivAssign<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    fn div_assign(&mut self, rhs: Matrix<T>) {
        *self = self as &Self / rhs;
    }
}

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
