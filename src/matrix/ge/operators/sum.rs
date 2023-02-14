use crate::{Matrix, Number};
use std::iter::Sum;

impl<T> Sum for Matrix<T>
where
    T: Number,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut iter = iter;
        let mut sum = iter.next().unwrap();
        for m in iter {
            sum += m;
        }
        sum
    }
}
