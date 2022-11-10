use super::SparseMatrix;
use crate::{matrix::ge::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(rhs: &Matrix<T>, lhs: &SparseMatrix<T>) -> Matrix<T>
where
    T: Number,
{
    if rhs.cols() != lhs.rows {
        panic!("Dimension mismatch.");
    }
    let mut new_matrix = Matrix::new(rhs.rows(), lhs.cols);

    for i in 0..rhs.rows() {
        for (&(j, k), &r) in lhs.elems.iter() {
            new_matrix[(k, i)] += rhs[(i, j)] * r;
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

        assert_eq!(ab[(0, 0)], 1.0);
        assert_eq!(ab[(1, 0)], 6.0);
        assert_eq!(ab[(1, 1)], 14.0);
    }
}
