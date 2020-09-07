use super::SparseMatrix;
use crate::{matrix::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(slf: &Matrix<T>, rhs: &SparseMatrix<T>) -> Matrix<T>
where
    T: Number,
{
    if slf.cols != rhs.rows {
        panic!("Dimension mismatch.");
    }
    let mut new_matrix = Matrix::new(slf.rows, rhs.cols);

    for i in 0..slf.rows() {
        for (&(j, k), &s) in rhs.elems.iter() {
            new_matrix[i][k] += s * slf[i][j];
        }
    }

    new_matrix
}

impl<T> Mul<SparseMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(&self, rhs)
    }
}

impl<T> Mul<SparseMatrix<T>> for &Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for &Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(self, rhs)
    }
}
