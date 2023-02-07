use super::SparseMatrix;
use crate::{matrix::ge::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(slf: &Matrix<T>, rhs: &SparseMatrix<T>) -> Matrix<T>
where
    T: Number,
{
    if slf.cols() != rhs.rows {
        panic!("Dimension mismatch.");
    }

    let s_rows = slf.cols();

    let new_matrix_vec = (0..s_rows)
        .map(|row| {
            (0..rhs.cols)
                .map(|col| {
                    let elems_orig = rhs
                        .elems
                        .iter()
                        .filter(|(&(_r_row, r_col), &_l)| r_col == col)
                        .map(|(&(r_row, _r_col), &l)| {
                            let elem = l * slf[(row, r_row)];
                            elem
                        })
                        .sum::<T>();
                    elems_orig
                })
                .collect::<Vec<T>>()
        })
        .collect::<Vec<Vec<T>>>()
        .concat();

    let new_matrix = Matrix::from(s_rows, new_matrix_vec).unwrap();

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

        println!("{:?}", ab)

        // assert_eq!(ab[(0, 0)], 1.0);
        // assert_eq!(ab[(1, 0)], 6.0);
        // assert_eq!(ab[(1, 1)], 14.0);
    }
}
