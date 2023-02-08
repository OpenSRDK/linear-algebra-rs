use crate::{sparse::SparseTensor, Number};
use std::ops::Add;

impl<T> Add for SparseTensor<T>
where
    T: Number,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
