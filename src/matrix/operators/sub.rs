use crate::matrix::Matrix;
use crate::number::Number;
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

fn sub<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if !lhs.same_size(rhs) {
        panic!("dimension mismatch")
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
        sub(rhs, self)
    }
}
