use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{generate_rank_combinations, RankCombinationId, TensorError};
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
use std::collections::HashMap;

pub trait DotProduct<T>
where
    T: Number,
{
    fn dot_product(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T>;
}

impl<'a, I, T> DotProduct<T> for I
where
    I: Iterator<Item = &'a SparseTensor<T>>,
    T: Number + 'a,
{
    fn dot_product(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T> {
        let tensors = self.collect::<Vec<_>>();
        let max_rank = tensors.iter().map(|t| t.rank()).max().unwrap();
        let mut new_sizes = vec![1; max_rank];

        for (i, t) in tensors.iter().enumerate() {
            for (j, &dim) in t.sizes.iter().enumerate() {
                if rank_combinations[i].get(&j).is_none() && dim > 1 {
                    if new_sizes[j] == 1 {
                        new_sizes[j] = dim;
                    } else {
                        panic!("The tensor whose a rank that is not aggregated and has a dimension greater than 1 can't be included.")
                    }
                }
            }
        }

        let mut result = SparseTensor::<T>::new(new_sizes);

        todo!();

        result
    }
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn dot(&self, rhs: &Self, rank_pairs: &[[RankIndex; 2]]) -> Self {
        let rank_combinations = generate_rank_combinations(rank_pairs);

        vec![self, rhs].into_iter().dot_product(&rank_combinations)
    }
}
