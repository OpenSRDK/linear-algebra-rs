use crate::matrix::Matrix;
use crate::number::Number;
use rayon::prelude::*;
use std::ops::Sub;

fn sub<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("dimension mismatch")
    }
    let mut lhs = lhs;

    lhs.elements
        .par_iter_mut()
        .zip(rhs.elements.par_iter())
        .map(|(l, &r)| {
            *l -= r;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T: Number> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        sub(self, &rhs)
    }
}

impl<T: Number> Sub<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<T: Number> Sub<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        sub(rhs, self)
    }
}
