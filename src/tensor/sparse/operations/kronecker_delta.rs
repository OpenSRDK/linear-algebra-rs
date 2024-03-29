use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::TensorError;
use crate::{sparse::SparseTensor, Number};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Mul;

impl<T> SparseTensor<T>
where
    T: Number,
{
    fn mul_kronecker_delta(&self, delta: &KroneckerDelta) -> Result<Self, TensorError> {
        let mut level_pair = (delta.0, delta.1);
        if level_pair.0 == level_pair.1 {
            return Ok(self.clone());
        }
        if level_pair.0 > level_pair.1 {
            let buffer = level_pair.0;
            level_pair.0 = level_pair.1;
            level_pair.1 = buffer;
        }

        let new_levels = self.rank().max(level_pair.1);
        let mut dims = self
            .sizes
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

    pub fn mul_kronecker_deltas(&self, deltas: &[KroneckerDelta]) -> Result<Self, TensorError> {
        let mut result = self.clone();
        for delta in deltas.iter() {
            result = result.mul_kronecker_delta(delta)?;
        }

        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KroneckerDelta(pub RankIndex, pub RankIndex);

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
        rhs.mul_kronecker_delta(&self).unwrap()
    }
}

impl<T> Mul<&SparseTensor<T>> for KroneckerDelta
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_delta(&self).unwrap()
    }
}

impl<T> Mul<KroneckerDelta> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta(&rhs).unwrap()
    }
}

impl<T> Mul<KroneckerDelta> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: KroneckerDelta) -> Self::Output {
        self.mul_kronecker_delta(&rhs).unwrap()
    }
}

impl<T> Mul<SparseTensor<T>> for Vec<KroneckerDelta>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_deltas(&self).unwrap()
    }
}

impl<T> Mul<&SparseTensor<T>> for Vec<KroneckerDelta>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: &SparseTensor<T>) -> Self::Output {
        rhs.mul_kronecker_deltas(&self).unwrap()
    }
}

impl<T> Mul<Vec<KroneckerDelta>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: Vec<KroneckerDelta>) -> Self::Output {
        self.mul_kronecker_deltas(&rhs).unwrap()
    }
}

impl<T> Mul<Vec<KroneckerDelta>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn mul(self, rhs: Vec<KroneckerDelta>) -> Self::Output {
        self.mul_kronecker_deltas(&rhs).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::sparse::SparseTensor;
    use crate::TensorError;

    #[test]
    fn test_kronecker_delta() -> Result<(), TensorError> {
        let mut a = SparseTensor::new(vec![2, 3, 4]);

        a[&[0, 0, 0]] = 1.0;
        a[&[1, 2, 3]] = 2.0;

        let mut b = SparseTensor::new(vec![2, 3, 4]);

        b[&[0, 0, 0]] = 1.0;
        b[&[1, 2, 3]] = 2.0;

        let c = a * KroneckerDelta(0, 1);
        let d = b * KroneckerDelta(1, 0);

        assert_eq!(c, d);

        Ok(())
    }
}
