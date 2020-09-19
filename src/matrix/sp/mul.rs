use super::SparseMatrix;
use crate::number::Number;
use std::ops::Mul;

fn mul<T>(slf: &SparseMatrix<T>, rhs: &SparseMatrix<T>) -> SparseMatrix<T>
where
    T: Number,
{
    if slf.cols != rhs.rows {
        panic!("Dimension mismatch.");
    }
    let mut new_matrix = SparseMatrix::new(slf.rows, rhs.cols);

    for (&(i, j), &s) in slf.elems.iter() {
        for (&(_, k), &r) in rhs.elems.iter().filter(|&(&(jr, _), _)| j == jr) {
            let sr = s * r;
            if sr == T::default() {
                continue;
            }

            *new_matrix.elems.entry((i, k)).or_insert(T::default()) += sr;
        }
    }

    new_matrix
}

impl<T> Mul<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(&self, rhs)
    }
}

impl<T> Mul<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(self, rhs)
    }
}
