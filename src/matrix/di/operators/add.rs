use crate::{
    number::{c64, Number},
    DiagonalMatrix, Matrix,
};
use rayon::prelude::*;
use std::ops::Add;

fn add_scalar<T>(lhs: DiagonalMatrix<T>, rhs: T) -> DiagonalMatrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.d.par_iter_mut().for_each(|l| {
        *l += rhs;
    });

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
        .for_each(|(l, &r)| {
            *l += r;
        });

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

fn add_mat<T>(lhs: Matrix<T>, rhs: &DiagonalMatrix<T>) -> Matrix<T>
where
    T: Number,
{
    let n = rhs.n();
    if lhs.rows() != n || lhs.cols() != n {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    for i in 0..n {
        lhs[i][i] += rhs[i];
    }

    lhs
}

impl<T> Add<Matrix<T>> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add_mat(rhs, &self)
    }
}

impl<T> Add<Matrix<T>> for &DiagonalMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add_mat(rhs, self)
    }
}

impl<T> Add<DiagonalMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: DiagonalMatrix<T>) -> Self::Output {
        add_mat(self, &rhs)
    }
}

impl<T> Add<&DiagonalMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &DiagonalMatrix<T>) -> Self::Output {
        add_mat(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn add() {
        let a = DiagonalMatrix::new(vec![2.0, 3.0]) + DiagonalMatrix::new(vec![4.0, 5.0]);
        assert_eq!(a[0], 6.0);
    }

    #[test]
    fn add_mat() {
        let a = DiagonalMatrix::new(vec![2.0, 3.0])
            + mat!(
              4.0, 5.0;
              6.0, 7.0
            );
        assert_eq!(a[(0, 0)], 6.0);
    }
}
