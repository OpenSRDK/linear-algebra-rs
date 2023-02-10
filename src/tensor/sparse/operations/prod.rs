use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{rank_combinations, RankCombinationId, TensorError};
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
use std::collections::HashMap;

pub trait InnerProd<T>
where
    T: Number,
{
    fn inner_prod(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T>;
}

impl<I, T> InnerProd<T> for I
where
    I: Iterator<Item = SparseTensor<T>>,
    T: Number,
{
    fn inner_prod(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T> {
        let tensors = self.collect::<Vec<_>>();
        let max_rank = tensors.iter().map(|t| t.rank()).max().unwrap();
        let mut new_dims = vec![1; max_rank];

        for (i, t) in tensors.iter().enumerate() {
            for (j, &dim) in t.sizes.iter().enumerate() {
                if rank_combinations[i].get(&j).is_none() && dim > 1 {
                    if new_dims[j] == 1 {
                        new_dims[j] = dim;
                    } else {
                        panic!("The tensor whose a rank that is not aggregated and has a dimension greater than 1 can't be included.")
                    }
                }
            }
        }

        let mut result = SparseTensor::<T>::new(new_dims);

        result
    }
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn inner_prod(self, rhs: Self, rank_pairs: &[[RankIndex; 2]]) -> Self {
        let rank_combinations = rank_combinations(rank_pairs);

        vec![self, rhs].into_iter().inner_prod(&rank_combinations)
    }
}
