use std::collections::HashMap;

use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;

pub trait TInnerProd<T>
where
    T: Number,
{
    fn inner_prod(self, rank_combinations: &[Vec<Option<String>>]) -> SparseTensor<T>;
}

impl<I, T> TInnerProd<T> for I
where
    I: Iterator<Item = SparseTensor<T>>,
    T: Number,
{
    fn inner_prod(self, rank_combinations: &[Vec<Option<String>>]) -> SparseTensor<T> {
        let tensors = self.collect::<Vec<_>>();
        let max_rank = tensors.iter().map(|t| t.rank()).max().unwrap();
        let mut new_dims = vec![1; max_rank];
        let mut identifier = HashMap::<String, (usize, usize)>::new();

        for (i, t) in tensors.iter().enumerate() {
            for (j, &dim) in t.dims.iter().enumerate() {
                if rank_combinations[i][j].is_none() && dim > 1 {
                    if new_dims[j] == 1 {
                        new_dims[j] = dim;
                    } else {
                        panic!("The tensor whose a rank that is not aggregated and has a dimension greater than 1 can't be included.")
                    }
                }

                if let Some(id) = &rank_combinations[i][j] {
                    identifier.insert(id.clone(), (i, j));
                }
            }
        }

        let mut result = SparseTensor::<T>::new(new_dims);

        result
    }
}
