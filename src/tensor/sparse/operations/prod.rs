use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

pub type CombinationId = String;

pub trait TInnerProd<T>
where
    T: Number,
{
    fn inner_prod(self, rank_combinations: &[HashMap<RankIndex, CombinationId>])
        -> SparseTensor<T>;
}

impl<I, T> TInnerProd<T> for I
where
    I: Iterator<Item = SparseTensor<T>>,
    T: Number,
{
    fn inner_prod(
        self,
        rank_combinations: &[HashMap<RankIndex, CombinationId>],
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
        let mut rank_combinations = vec![HashMap::new(); 2];
        for rank_pair in rank_pairs.iter() {
            let id = thread_rng().gen::<u32>();
            rank_combinations[0].insert(rank_pair[0], id.to_string());
            rank_combinations[1].insert(rank_pair[1], id.to_string());
        }

        vec![self, rhs].into_iter().inner_prod(&rank_combinations)
    }
}
