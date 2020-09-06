use super::SparseMatrix;
use crate::{matrix::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(slf: &SparseMatrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if slf.cols != rhs.rows {
        panic!("dimension mismatch");
    }
    let mut new_matrix = Matrix::new(slf.rows, rhs.cols);

    for (&(i, j), &s) in slf.elems.iter() {
        for k in 0..rhs.cols {
            new_matrix[i][k] += s * rhs[j][k];
        }
    }

    new_matrix
}

impl<T> Mul<Matrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        mul(&self, rhs)
    }
}

impl<T> Mul<Matrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        mul(self, &rhs)
    }
}

impl<T> Mul<&Matrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        mul(self, rhs)
    }
}