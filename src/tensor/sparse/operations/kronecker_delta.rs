use crate::tensor::Tensor;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Mul;

impl<T> SparseTensor<T>
where
    T: Number,
{
    fn mul_kronecker_delta(&self, delta: &KroneckerDelta) -> Self {
        let mut level_pair = (delta.0, delta.1);
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

    pub fn mul_kronecker_deltas(&self, deltas: &[KroneckerDelta]) -> Self {
        let mut result = self.clone();
        for delta in deltas.iter() {
            result = result.mul_kronecker_delta(delta);
        }

        result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KroneckerDelta(pub usize, pub usize);

impl Mul for KroneckerDelta {
    type Output = Vec<KroneckerDelta>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        vec![self, rhs]
    }
}

impl Mul<Vec<KroneckerDelta>> for KroneckerDelta {
    type Output = Vec<KroneckerDelta>;

    fn mul(self, rhs: Vec<KroneckerDelta>) -> Self::Output {
        let mut result = rhs;
        result.push(self);
        result
    }
}

impl Mul<KroneckerDelta> for Vec<KroneckerDelta> {
    type Output = Vec<KroneckerDelta>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        let mut result = self;
        result.push(rhs);
        result
    }
}

impl<T> Mul<SparseTensor<T>> for KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta(&self)
    }
}

impl<T> Mul<&SparseTensor<T>> for KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta(&self)
    }
}

impl<T> Mul<KroneckerDelta> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta(&rhs)
    }
}

impl<T> Mul<KroneckerDelta> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta(&rhs)
    }
}

impl<T> Mul<SparseTensor<T>> for Vec<KroneckerDelta>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_deltas(&self)
    }
}

impl<T> Mul<&SparseTensor<T>> for Vec<KroneckerDelta>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_deltas(&self)
    }
}

impl<T> Mul<Vec<KroneckerDelta>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: Vec<KroneckerDelta>) -> Self::Output {
        self.mul_kronecker_deltas(&rhs)
    }
}

impl<T> Mul<Vec<KroneckerDelta>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: Vec<KroneckerDelta>) -> Self::Output {
        self.mul_kronecker_deltas(&rhs)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::sparse::SparseTensor;
    use crate::TensorError;

    #[test]
    fn test_kronecker_delta_mul() -> Result<(), TensorError> {
        let a =
            KroneckerDelta(0, 1) * KroneckerDelta(1, 2) * SparseTensor::<f64>::new(vec![2, 2, 2]);
        let b = KroneckerDelta(1, 2) * KroneckerDelta(0, 1);

        Ok(())
    }
}
