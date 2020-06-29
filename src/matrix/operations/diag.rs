use crate::{
    matrix::Matrix,
    number::Number,
    types::{Diagonal, Square},
};

pub fn diag<U>(vec: &[U]) -> Matrix<Diagonal, U>
where
    U: Number,
{
    let n = vec.len();
    let mut new_matrix = Matrix::<Square, U>::zeros(n).transmute();
    for i in 0..n {
        new_matrix[i][i] = vec[i];
    }

    new_matrix
}
