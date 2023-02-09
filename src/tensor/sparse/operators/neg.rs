use crate::{sparse::SparseTensor, Number};
use std::ops::Neg;

impl<T> Neg for SparseTensor<T>
where
    T: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
    }
}
