use crate::matrix::ge::Matrix;
use crate::number::{c64, Number};
use rayon::prelude::*;
use std::ops::Add;

fn add_scalar<T>(lhs: Matrix<T>, rhs: T) -> Matrix<T>
where
    T: Number,
{
    let mut lhs = lhs;

    lhs.elems
        .par_iter_mut()
        .map(|l| {
            *l += rhs;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Add<T> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: T) -> Self::Output {
        add_scalar(self, rhs)
    }
}

impl Add<Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        add_scalar(rhs, self)
    }
}

impl Add<Matrix<c64>> for c64 {
    type Output = Matrix<c64>;

    fn add(self, rhs: Matrix<c64>) -> Self::Output {
        add_scalar(rhs, self)
    }
}

fn add<T>(lhs: Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    if !lhs.same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    lhs.elems
        .par_iter_mut()
        .zip(rhs.elems.par_iter())
        .map(|(l, &r)| {
            *l += r;
        })
        .collect::<Vec<_>>();

    lhs
}

impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add(self, &rhs)
    }
}

impl<T> Add<&Matrix<T>> for Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        add(self, rhs)
    }
}

impl<T> Add<Matrix<T>> for &Matrix<T>
where
    T: Number,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        add(rhs, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) + mat!(
            5.0, 6.0;
            7.0, 8.0
        );
        assert_eq!(a[(0, 0)], 6.0);
        assert_eq!(a[(0, 1)], 8.0);
        assert_eq!(a[(1, 0)], 10.0);
        assert_eq!(a[(1, 1)], 12.0);
    }
}
