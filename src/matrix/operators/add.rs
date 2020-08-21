use crate::matrix::Matrix;
use crate::number::Number;
use rayon::prelude::*;
use std::ops::Add;

fn add<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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
            *l += r;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T: Number> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add(self, &rhs)
    }
}

impl<T: Number> Add<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        add(self, rhs)
    }
}

impl<T: Number> Add<Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add(rhs, self)
    }
}
