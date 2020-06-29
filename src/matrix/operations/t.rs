use crate::matrix::Matrix;
use crate::number::Number;
use crate::types::*;

fn t<T, U, V>(slf: &Matrix<T, U>) -> Matrix<V, U>
where
    T: Type,
    U: Number,
    V: Type,
{
    let mut new_matrix = Matrix::<Standard, U>::zeros(slf.columns, slf.rows).transmute();

    for i in 0..new_matrix.rows {
        for j in 0..new_matrix.columns {
            new_matrix[i][j] = slf[j][i];
        }
    }

    new_matrix
}

macro_rules! implement {
    {$t1: ty, $t2: ty} => {
        impl<U> Matrix<$t1, U>
        where
            U: Number,
        {
            pub fn t(&self) -> Matrix<$t2, U> {
                t(self)
            }
        }
    };
}

implement! {Standard, Standard}
implement! {Square, Square}
implement! {UpperTriangle, LowerTriangle}
implement! {LowerTriangle, UpperTriangle}

macro_rules! implement_clone {
    {$t1: ty} => {
        impl<U> Matrix<$t1, U>
        where
            U: Number,
        {
            pub fn t(&self) -> Matrix<$t1, U> {
                self.clone()
            }
        }
    };
}

implement_clone! {Diagonal}
implement_clone! {PositiveDefinite}
implement_clone! {PositiveSemiDefinite}
