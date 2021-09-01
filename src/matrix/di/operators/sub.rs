use crate::{
    number::{c64, Number},
    DiagonalMatrix,
};
use rayon::prelude::*;
use std::ops::Sub;

fn sub_scalar<T>(lhs: DiagonalMatrix<T>, rhs: T) -> DiagonalMatrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.d
        .par_iter_mut()
        .map(|l| {
            *l -= rhs;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Sub<T> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn sub(self, rhs: T) -> Self::Output {
        sub_scalar(self, rhs)
    }
}

impl Sub<DiagonalMatrix> for f64 {
    type Output = DiagonalMatrix;

    fn sub(self, rhs: DiagonalMatrix) -> Self::Output {
        sub_scalar(rhs, self)
    }
}

impl Sub<DiagonalMatrix<c64>> for c64 {
    type Output = DiagonalMatrix<c64>;

    fn sub(self, rhs: DiagonalMatrix<c64>) -> Self::Output {
        sub_scalar(rhs, self)
    }
}

fn sub<T>(lhs: DiagonalMatrix<T>, rhs: &DiagonalMatrix<T>) -> DiagonalMatrix<T>
where
    T: Number,
{
    if lhs.n() != rhs.n() {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    lhs.d
        .par_iter_mut()
        .zip(rhs.d.par_iter())
        .map(|(l, &r)| {
            *l -= r;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Sub<DiagonalMatrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn sub(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        sub(self, &rhs)
    }
}

impl<T> Sub<&DiagonalMatrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn sub(self, rhs: &DiagonalMatrix<T>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<T> Sub<DiagonalMatrix<T>> for &DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn sub(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        sub(rhs, self)
    }
}
