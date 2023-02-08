use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn inner_prod(
        &self,
        rhs: Self,
        level_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, TensorError> {
        for &(level, level_prime) in &level_pairs {
            if self.dim(level) != rhs.dim(level_prime) {
                return Err(TensorError::DimensionMismatch);
            }
        }

        let mut new_dims1 = self.dims.clone();
        let mut new_dims2 = rhs.dims.clone();
        level_pairs.iter().for_each(|(level, level_prime)| {
            new_dims1.remove(*level);
            new_dims2.remove(*level_prime);
        });

        let new = Self::new([new_dims1, new_dims2].concat());

        todo!();

        Ok(new)
    }
}
