use crate::matrix::Matrix;
use crate::number::Number;
use rayon::prelude::*;
use std::ops::Sub;

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
