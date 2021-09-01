use crate::{
    number::{c64, Number},
    DiagonalMatrix,
};
use rayon::prelude::*;
use std::ops::Add;

fn add_scalar<T>(lhs: DiagonalMatrix<T>, rhs: T) -> DiagonalMatrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.d
        .par_iter_mut()
        .map(|l| {
            *l += rhs;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Add<T> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn add(self, rhs: T) -> Self::Output {
        add_scalar(self, rhs)
    }
}

impl Add<DiagonalMatrix> for f64 {
    type Output = DiagonalMatrix;

    fn add(self, rhs: DiagonalMatrix) -> Self::Output {
        add_scalar(rhs, self)
    }
}

impl Add<DiagonalMatrix<c64>> for c64 {
    type Output = DiagonalMatrix<c64>;

    fn add(self, rhs: DiagonalMatrix<c64>) -> Self::Output {
        add_scalar(rhs, self)
    }
}

fn add<T>(lhs: DiagonalMatrix<T>, rhs: &DiagonalMatrix<T>) -> DiagonalMatrix<T>
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
            *l += r;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Add<DiagonalMatrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn add(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        add(self, &rhs)
    }
}

impl<T> Add<&DiagonalMatrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn add(self, rhs: &DiagonalMatrix<T>) -> Self::Output {
        add(self, rhs)
    }
}

impl<T> Add<DiagonalMatrix<T>> for &DiagonalMatrix<T>
where
    T: Number,
{
    type Output = DiagonalMatrix<T>;

    fn add(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        add(rhs, self)
    }
}
