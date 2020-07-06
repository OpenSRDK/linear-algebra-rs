use crate::matrix::Matrix;
use crate::number::c64;
use crate::types::*;

fn adjoint<T, V>(slf: &Matrix<T, c64>) -> Matrix<V, c64>
where
    T: Type,
    V: Type,
{
    let mut new_matrix = Matrix::<Standard, c64>::zeros(slf.columns, slf.rows).transmute();

    for i in 0..new_matrix.rows {
        for j in 0..new_matrix.columns {
            new_matrix[i][j] = c64::new(slf[j][i].re, -1.0 * slf[j][i].im);
        }
    }

    new_matrix
}

macro_rules! implement {
    {$t1: ty, $t2: ty} => {
        impl Matrix<$t1, c64>
        {
            /// # Transpose
            pub fn adjoint(&self) -> Matrix<$t2, c64> {
                adjoint(self)
            }
        }
    };
}

implement! {Standard, Standard}
implement! {Square, Square}
implement! {UpperTriangle, LowerTriangle}
implement! {LowerTriangle, UpperTriangle}
implement! {Diagonal, Diagonal}
implement! {PositiveDefinite, Square}
implement! {PositiveSemiDefinite, Square}
