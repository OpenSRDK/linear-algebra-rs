use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn inner_prod(&self, rhs: &Self, level_pairs: &[(usize, usize)]) -> Self {
        for &(level, level_prime) in level_pairs {
            if self.dim(level) != rhs.dim(level_prime) {
                panic!("Dimension mismatch.")
            }
        }

        let mut new_dims1 = self.dims.clone();
        let mut new_dims2 = rhs.dims.clone();
        level_pairs.iter().for_each(|(level, level_prime)| {
            new_dims1[*level] = 1;
            new_dims2[*level_prime] = 1;
        });

        // Convert to matrix format to calculate rapidly
        for &level_pair in level_pairs {}

        todo!();
    }
}
