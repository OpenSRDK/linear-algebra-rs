//これも掛け算逆っぽい
use super::SparseMatrix;
use crate::{matrix::ge::Matrix, number::Number};
use std::ops::Mul;

fn mul<T>(slf: &SparseMatrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if slf.cols != rhs.rows() {
        panic!("Dimension mismatch.");
    }

    let r_cols = rhs.cols();

    let new_matrix_vec = (0..r_cols)
        .map(|col| {
            (0..slf.rows)
                .map(|row| {
                    let elems_orig = slf
                        .elems
                        .iter()
                        .filter(|(&(s_row, _s_col), &_l)| s_row == row)
                        .map(|(&(_s_row, s_col), &l)| {
                            let elem = l * rhs[(s_col, col)];
                            elem
                        })
                        .sum::<T>();
                    elems_orig
                })
                .collect::<Vec<T>>()
        })
        .collect::<Vec<Vec<T>>>()
        .concat();

    let new_matrix = Matrix::from(slf.rows, new_matrix_vec).unwrap();

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
    #[test]
    fn it_works() {
        let mut a = SparseMatrix::new(3, 3);
        a[(0, 1)] = 1.0;
        a[(2, 0)] = 2.0;
        a[(2, 2)] = 1.0;
        let b = mat![
            1.0, 2.0;
            3.0, 4.0;
            5.0, 6.0
        ];
        let c = a * b;

        println!("{:?}", c);

        //assert_eq!(c[(0, 0)], 5.0);
        //assert_eq!(c[(1, 1)], 18.0);
    }
}
