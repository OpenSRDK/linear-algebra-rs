use std::collections::HashMap;

use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;

pub trait TInnerProd<T>
where
    T: Number,
{
    fn inner_prod(self, rank_combinations: &[HashMap<usize, String>]) -> SparseTensor<T>;
}

impl<I, T> TInnerProd<T> for I
where
    I: Iterator<Item = SparseTensor<T>>,
    T: Number,
{
    fn inner_prod(self, rank_combinations: &[HashMap<usize, String>]) -> SparseTensor<T> {
        let tensors = self.collect::<Vec<_>>();
        let max_rank = tensors.iter().map(|t| t.rank()).max().unwrap();
        let mut new_dims = vec![1; max_rank];

        for (i, t) in tensors.iter().enumerate() {
            for (j, &dim) in t.dims.iter().enumerate() {
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
    pub fn inner_prod(self, rhs: Self, rank_pairs: &[(usize, usize)]) -> Self {
        let mut rank_combinations = vec![HashMap::new(); 2];
        for (i, rank_pair) in rank_pairs.iter().enumerate() {
            rank_combinations[0].insert(rank_pair.0, i.to_string());
            rank_combinations[1].insert(rank_pair.1, i.to_string());
        }

        vec![self, rhs].into_iter().inner_prod(&rank_combinations)
    }
}
