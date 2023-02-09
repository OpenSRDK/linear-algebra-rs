use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Mul;

impl<T> SparseTensor<T>
where
    T: Number,
{
    fn mul_kronecker_delta(&self, mut level_pair: (usize, usize)) -> Self {
        if level_pair.0 == level_pair.1 {
            return self.clone();
        }
        if level_pair.0 > level_pair.1 {
            let buffer = level_pair.0;
            level_pair.0 = level_pair.1;
            level_pair.1 = buffer;
        }

        let new_levels = self.rank().max(level_pair.1);
        let mut dims = self
            .dims
            .iter()
            .cloned()
            .chain((0..new_levels - self.rank()).map(|_| 1usize))
            .collect::<Vec<_>>();

        dims[level_pair.1] = dims[level_pair.0];
        dims[level_pair.0] = 1;

        let new_elems = self
            .elems
            .par_iter()
            .map(|(indices, value)| {
                let mut new_indices = indices
                    .iter()
                    .cloned()
                    .chain((0..new_levels - self.rank()).map(|_| 0usize))
                    .collect::<Vec<_>>();

                new_indices[level_pair.1] = new_indices[level_pair.0];
                new_indices[level_pair.0] = 0;

                (new_indices, *value)
            })
            .collect::<HashMap<_, _>>();

        Self::from(dims, new_elems)
    }

    pub fn mul_kronecker_deltas(&self, level_pairs: &[(usize, usize)]) -> Self {
        let mut result = self.clone();
        for level_pair in level_pairs {
            result = result.mul_kronecker_delta(*level_pair);
        }

        result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KroneckerDelta(pub usize, pub usize);

impl<T> Mul<SparseTensor<T>> for KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta((self.0, self.1))
    }
}

impl<T> Mul<&SparseTensor<T>> for KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta((self.0, self.1))
    }
}

impl<T> Mul<SparseTensor<T>> for &KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta((self.0, self.1))
    }
}

impl<T> Mul<&SparseTensor<T>> for &KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta((self.0, self.1))
    }
}

impl<T> Mul<KroneckerDelta> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta((rhs.0, rhs.1))
    }
}

impl<T> Mul<&KroneckerDelta> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta((rhs.0, rhs.1))
    }
}

impl<T> Mul<KroneckerDelta> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta((rhs.0, rhs.1))
    }
}

impl<T> Mul<&KroneckerDelta> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta((rhs.0, rhs.1))
    }
}
