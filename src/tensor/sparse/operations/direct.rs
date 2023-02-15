use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
use std::collections::HashMap;

pub trait DirectProduct<T>
where
    T: Number,
{
    fn direct_product(self) -> SparseTensor<T>;
}

impl<'a, I, T> DirectProduct<T> for I
where
    I: Iterator<Item = &'a SparseTensor<T>>,
    T: Number + 'a,
{
    fn direct_product(self) -> SparseTensor<T> {
        todo!();
    }
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn direct(&self, rhs: &Self) -> Self {
        vec![self, rhs].into_iter().direct_product()
    }
}
