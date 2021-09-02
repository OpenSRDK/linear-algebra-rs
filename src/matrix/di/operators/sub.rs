use crate::{
    number::{c64, Number},
    DiagonalMatrix, Matrix,
};
use rayon::prelude::*;
use std::ops::Sub;

fn sub_scalar<T>(lhs: DiagonalMatrix<T>, rhs: T) -> DiagonalMatrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.d.par_iter_mut().for_each(|l| {
        *l -= rhs;
    });

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
        -sub_scalar(rhs, self)
    }
}

impl Sub<DiagonalMatrix<c64>> for c64 {
    type Output = DiagonalMatrix<c64>;

    fn sub(self, rhs: DiagonalMatrix<c64>) -> Self::Output {
        -sub_scalar(rhs, self)
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
        .for_each(|(l, &r)| {
            *l -= r;
        });

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
        -sub(rhs, self)
    }
}

fn sub_mat<T>(lhs: Matrix<T>, rhs: &DiagonalMatrix<T>) -> Matrix<T>
where
    T: Number,
{
    let n = rhs.n();
    if lhs.rows() != n || lhs.cols() != n {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    for i in 0..n {
        lhs[i][i] -= rhs[i];
    }

    lhs
}

impl<T> Sub<Matrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        -sub_mat(rhs, &self)
    }
}

impl<T> Sub<Matrix<T>> for &DiagonalMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        -sub_mat(rhs, self)
    }
}

impl<T> Sub<DiagonalMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        sub_mat(self, &rhs)
    }
}

impl<T> Sub<&DiagonalMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &DiagonalMatrix<T>) -> Self::Output {
        sub_mat(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn sub() {
        let a = DiagonalMatrix::new(vec![2.0, 3.0]) - DiagonalMatrix::new(vec![4.0, 5.0]);
        assert_eq!(a[0], -2.0);
    }

    #[test]
    fn sub_mat() {
        let a = DiagonalMatrix::new(vec![2.0, 3.0])
            - mat!(
              4.0, 5.0;
              6.0, 7.0
            );
        assert_eq!(a[(0, 0)], -2.0);
    }
}
