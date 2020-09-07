use super::SparseMatrix;
use crate::{matrix::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(slf: &SparseMatrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if slf.cols != rhs.rows {
        panic!("Dimension mismatch.");
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

#[cfg(test)]
mod tests {
    use crate::*;
    use std::collections::HashMap;
    #[test]
    fn it_works() {
        let mut a = SparseMatrix::new(2, 3, HashMap::new());
        a[(0, 0)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(1, 2)] = 3.0;
        let b = mat![
            1.0, 3.0;
            2.0, 4.0;
            3.0, 6.0
        ];
        let c = a * b;

        assert_eq!(c[0][0], 5.0);
        assert_eq!(c[1][1], 18.0);
    }
}
