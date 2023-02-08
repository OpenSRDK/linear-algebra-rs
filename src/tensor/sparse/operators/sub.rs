use crate::{sparse::SparseTensor, Number};
use std::ops::Sub;

impl<T> Sub for SparseTensor<T>
where
    T: Number,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
