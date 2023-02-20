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
        let terms = self.collect::<Vec<_>>();
        let new_sizes = terms.iter().fold(vec![], |mut acc, &next| {
            if acc.len() < next.sizes.len() {
                for i in 0..acc.len() {
                    acc[i] *= next.size(i);
                }
                acc.extend(next.sizes[acc.len()..].iter().copied());
            } else {
                for i in 0..next.sizes.len() {
                    acc[i] *= next.size(i);
                }
            }
            acc
        });

        let mut result = SparseTensor::<T>::new(new_sizes);

        todo!();

        result
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
