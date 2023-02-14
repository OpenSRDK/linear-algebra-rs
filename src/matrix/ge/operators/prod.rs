use crate::{Matrix, Number};
use std::iter::Product;

impl<T> Product for Matrix<T>
where
    T: Number,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut iter = iter;
        let mut product = iter.next().unwrap();
        for m in iter {
            product *= m;
        }
        product
    }
}
