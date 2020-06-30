use crate::matrix::Matrix;
use crate::number::Number;
use crate::types::*;
use rayon::prelude::*;
use std::ops::Add;

fn add<T, U, V, W>(mut slf: Matrix<T, U>, rhs: Matrix<V, U>) -> Matrix<W, U>
where
    T: Type,
    U: Number,
    V: Type,
    W: Type,
{
    if !slf.is_same_size(&rhs) {
        panic!("dimension mismatch")
    }

    slf.elements
        .par_iter_mut()
        .zip(rhs.elements.into_par_iter())
        .map(|(s, r)| {
            *s += r;
        })
        .collect::<Vec<_>>();

    slf.transmute()
}

macro_rules! implement {
    ($t1: ty, $t2: ty, $t3: ty) => {
        impl<U: Number> Add<Matrix<$t2, U>> for Matrix<$t1, U> {
            type Output = Matrix<$t3, U>;

            fn add(self, rhs: Matrix<$t2, U>) -> Self::Output {
                add(self, rhs)
            }
        }
    };
}

macro_rules! implement_commutate {
    ($t1: ty, $t2: ty, $t3: ty) => {
        implement! {$t1, $t2, $t3}
        implement! {$t2, $t1, $t3}
    };
}

implement! {Standard, Standard, Standard}
implement_commutate! {Standard, Square, Standard}
implement_commutate! {Standard, UpperTriangle, Standard}
implement_commutate! {Standard, LowerTriangle, Standard}
implement_commutate! {Standard, Diagonal, Standard}
implement_commutate! {Standard, PositiveDefinite, Standard}
implement_commutate! {Standard, PositiveSemiDefinite, Standard}

implement! {Square, Square, Square}
implement_commutate! {Square, UpperTriangle, Square}
implement_commutate! {Square, LowerTriangle, Square}
implement_commutate! {Square, Diagonal, Square}
implement_commutate! {Square, PositiveDefinite, Square}
implement_commutate! {Square, PositiveSemiDefinite, Square}

implement! {UpperTriangle, UpperTriangle, UpperTriangle}
implement_commutate! {UpperTriangle, LowerTriangle, Square}
implement_commutate! {UpperTriangle, Diagonal, UpperTriangle}
implement_commutate! {UpperTriangle,PositiveDefinite, Square}
implement_commutate! {UpperTriangle, PositiveSemiDefinite, Square}

implement! {LowerTriangle, LowerTriangle, LowerTriangle}
implement_commutate! {LowerTriangle, Diagonal, LowerTriangle}
implement_commutate! {LowerTriangle, PositiveDefinite, Square}
implement_commutate! {LowerTriangle, PositiveSemiDefinite, Square}

implement! {Diagonal, PositiveDefinite, Square}
implement_commutate! {Diagonal, PositiveSemiDefinite, Square}

implement! {PositiveDefinite, PositiveDefinite, Square}
implement_commutate! {PositiveDefinite, PositiveSemiDefinite, Square}

implement! {PositiveSemiDefinite, PositiveSemiDefinite, Square}
