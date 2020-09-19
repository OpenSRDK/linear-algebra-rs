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
        for (&(j, k), &r) in rhs.elems.iter() {
            new_matrix[i][k] += slf[i][j] * r;
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

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let b = SparseMatrix::from(
            2,
            2,
            vec![
                ((0usize, 0usize), 1.0),
                ((0usize, 1usize), 2.0),
                ((1usize, 1usize), 2.0),
            ]
            .into_iter()
            .collect(),
        );
        let ab = a * b;

        assert_eq!(ab[0][0], 1.0);
        assert_eq!(ab[0][1], 6.0);
        assert_eq!(ab[1][1], 14.0);
    }
}
