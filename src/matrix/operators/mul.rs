use crate::matrix::Matrix;
use crate::number::{c64, Number};
use crate::types::*;
use blas::{dgemm, zgemm};
use rayon::prelude::*;

use std::{intrinsics::transmute, ops::Mul};

fn mul<T, U>(slf: U, mut rhs: Matrix<T, U>) -> Matrix<T, U>
where
    T: Type,
    U: Number,
{
    rhs.elements
        .par_iter_mut()
        .map(|r| {
            *r *= slf;
        })
        .collect::<Vec<_>>();

    rhs
}

impl<T> Mul<Matrix<T>> for f64
where
    T: Type,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        mul(self, rhs)
    }
}

impl<T> Mul<Matrix<T, c64>> for c64
where
    T: Type,
{
    type Output = Matrix<T, c64>;

    fn mul(self, rhs: Matrix<T, c64>) -> Self::Output {
        mul(self, rhs)
    }
}

fn mul_f64<U, V, W>(lhs: &Matrix<U>, rhs: &Matrix<V>) -> Matrix<W>
where
    U: Type,
    V: Type,
    W: Type,
{
    if lhs.columns != rhs.rows {
        panic!("dimension mismatch")
    }

    let mut new_matrix = Matrix::<Standard>::zeros(lhs.rows, rhs.columns).transmute();

    unsafe {
        dgemm(
            'N' as u8,
            'N' as u8,
            lhs.rows as i32,
            lhs.columns as i32,
            rhs.columns as i32,
            1.0,
            rhs.elements.as_slice(),
            lhs.rows as i32,
            lhs.elements.as_slice(),
            rhs.columns as i32,
            0.0,
            &mut new_matrix.elements,
            lhs.rows as i32,
        );
    }

    new_matrix
}

fn mul_c64<U, V, W>(lhs: &Matrix<U, c64>, rhs: &Matrix<V, c64>) -> Matrix<W, c64>
where
    U: Type,
    V: Type,
    W: Type,
{
    if lhs.columns != rhs.rows {
        panic!("dimension mismatch")
    }

    let mut new_matrix = Matrix::<Standard, c64>::zeros(lhs.rows, rhs.columns).transmute();

    unsafe {
        zgemm(
            'N' as u8,
            'N' as u8,
            lhs.rows as i32,
            lhs.columns as i32,
            rhs.columns as i32,
            blas::c64::new(1.0, 0.0),
            transmute::<&[c64], &[blas::c64]>(&rhs.elements),
            lhs.rows as i32,
            transmute::<&[c64], &[blas::c64]>(&lhs.elements),
            rhs.columns as i32,
            blas::c64::new(1.0, 0.0),
            transmute::<&mut [c64], &mut [blas::c64]>(&mut new_matrix.elements),
            lhs.rows as i32,
        );
    }

    new_matrix
}

impl<T, U> Mul<U> for Matrix<T, U>
where
    T: Type,
    U: Number,
{
    type Output = Self;

    fn mul(self, rhs: U) -> Self::Output {
        mul(rhs, self)
    }
}

macro_rules! implement_types {
    ($t1: ty, $t2: ty, $t3: ty, $t4: ty, $e: expr) => {
        impl Mul<Matrix<$t2, $t4>> for Matrix<$t1, $t4> {
            type Output = Matrix<$t3, $t4>;

            fn mul(self, rhs: Matrix<$t2, $t4>) -> Self::Output {
                $e(&self, &rhs)
            }
        }

        impl Mul<&Matrix<$t2, $t4>> for Matrix<$t1, $t4> {
            type Output = Matrix<$t3, $t4>;

            fn mul(self, rhs: &Matrix<$t2, $t4>) -> Self::Output {
                $e(&self, &rhs)
            }
        }

        impl Mul<Matrix<$t2, $t4>> for &Matrix<$t1, $t4> {
            type Output = Matrix<$t3, $t4>;

            fn mul(self, rhs: Matrix<$t2, $t4>) -> Self::Output {
                $e(self, &rhs)
            }
        }

        impl Mul<&Matrix<$t2, $t4>> for &Matrix<$t1, $t4> {
            type Output = Matrix<$t3, $t4>;

            fn mul(self, rhs: &Matrix<$t2, $t4>) -> Self::Output {
                $e(self, rhs)
            }
        }
    };
}

macro_rules! implement {
    ($t1: ty, $t2: ty, $t3: ty) => {
        implement_types! {$t1, $t2, $t3, f64, mul_f64}
        implement_types! {$t1, $t2, $t3, c64, mul_c64}
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

implement! {Diagonal, Diagonal, Diagonal}
implement_commutate! {Diagonal, PositiveDefinite, Square}
implement_commutate! {Diagonal, PositiveSemiDefinite, Square}

implement! {PositiveDefinite, PositiveDefinite, Square}
implement_commutate! {PositiveDefinite, PositiveSemiDefinite, Square}

implement! {PositiveSemiDefinite, PositiveSemiDefinite, Square}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = mat!(
          1.0, 2.0;
          3.0, 4.0
        ) * mat!(
          5.0, 6.0;
          7.0, 8.0
        );
        assert_eq!(a[0][0], 19.0)
    }
}
