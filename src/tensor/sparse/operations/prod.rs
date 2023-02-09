use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn inner_prod(&self, rhs: &Self, rank_pairs: &[(usize, usize)]) -> Self {
        for &rank_pair in rank_pairs {
            if self.dim(rank_pair.0) != rhs.dim(rank_pair.1) {
                panic!("Dimension mismatch.")
            }
        }

        let mut new_dims1 = self.dims.clone();
        let mut new_dims2 = rhs.dims.clone();
        rank_pairs.iter().for_each(|&rank_pair| {
            new_dims1[rank_pair.0] = 1;
            new_dims2[rank_pair.1] = 1;
        });

        // Convert to matrix format to calculate rapidly
        for &rank_pair in rank_pairs {}

        todo!();
    }
}
