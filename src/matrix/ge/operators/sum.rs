use crate::{Matrix, Number};
use std::iter::Sum;

impl<T> Sum for Matrix<T>
where
    T: Number,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap()
    }
}
