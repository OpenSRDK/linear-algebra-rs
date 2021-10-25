use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::Sub;

fn sub_scalar<T>(lhs: Matrix<T>, rhs: T) -> Matrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.elems
        .par_iter_mut()
        .map(|l| {
            *l -= rhs;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Sub<T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: T) -> Self::Output {
        sub_scalar(self, rhs)
    }
}

impl Sub<Matrix> for f64 {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        -sub_scalar(rhs, self)
    }
}

impl Sub<Matrix<c64>> for c64 {
    type Output = Matrix<c64>;

    fn sub(self, rhs: Matrix<c64>) -> Self::Output {
        -sub_scalar(rhs, self)
    }
}

fn sub<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if !lhs.same_size(rhs) {
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
