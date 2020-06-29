use crate::{
    matrix::Matrix,
    number::Number,
    types::{Diagonal, Square},
};

/// # Identity
pub fn identity<U>(n: usize) -> Matrix<Diagonal, U>
where
    U: Number,
{
    let mut new_matrix = Matrix::<Square, U>::zeros(n).transmute();
    for i in 0..n {
        new_matrix[i][i] = U::one();
    }

    new_matrix
}
