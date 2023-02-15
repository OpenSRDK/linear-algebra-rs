pub mod operations;
pub mod operators;

pub use operations::*;

use crate::{Matrix, Number, RankIndex, Tensor, TensorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseTensor<T = f64>
where
    T: Number,
{
    sizes: Vec<usize>,
    elems: HashMap<Vec<usize>, T>,
    default: T,
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn new(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            elems: HashMap::new(),
            default: T::default(),
        }
    }

    pub fn from(sizes: Vec<usize>, elems: HashMap<Vec<usize>, T>) -> Result<Self, TensorError> {
        for (index, _) in elems.iter() {
            if index.len() != sizes.len() {
                return Err(TensorError::RankMismatch);
            }
            for (rank, &d) in index.iter().enumerate() {
                if sizes[rank] <= d {
                    return Err(TensorError::OutOfRange);
                }
            }
        }
        Ok(Self {
            sizes,
            elems,
            default: T::default(),
        })
    }

    pub fn is_same_size(&self, other: &SparseTensor<T>) -> bool {
        self.sizes == other.sizes
    }

    pub fn total_size(&self) -> usize {
        self.sizes.iter().product()
    }

    pub fn not_1dimension_ranks(&self) -> usize {
        self.sizes.iter().filter(|&d| *d != 1).count()
    }

    pub fn reduce_1dimension_rank(&self) -> Self {
        let mut new_dims = vec![];
        for d in self.sizes.iter() {
            if *d != 1 {
                new_dims.push(*d);
            }
        }

        // TODO: parallelize
        let mut new_elems = HashMap::new();
        for (k, v) in self.elems.iter() {
            let mut new_k = vec![];
            for (i, d) in k.iter().enumerate() {
                if self.sizes[i] != 1 {
                    new_k.push(*d);
                }
            }
            new_elems.insert(new_k, *v);
        }

        Self {
            sizes: new_dims,
            elems: new_elems,
            default: self.default,
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        if self.rank() != 1 {
            panic!("SparseTensor::to_vec() is only available for rank 1 tensor.");
        }

        let mut vec = vec![T::default(); self.sizes[0]];
        for (k, v) in self.elems.iter() {
            vec[k[0]] = *v;
        }

        vec
    }

    pub fn to_mat(&self) -> Matrix<T> {
        if self.rank() != 2 {
            panic!("SparseTensor::to_mat() is only available for rank 2 tensor.");
        }

        let mut mat = Matrix::new(self.sizes[0], self.sizes[1]);
        for (k, v) in self.elems.iter() {
            mat[(k[0], k[1])] = *v;
        }
        mat
    }

    pub fn elems(&self) -> &HashMap<Vec<usize>, T> {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut HashMap<Vec<usize>, T> {
        &mut self.elems
    }
}

impl<T> Tensor<T> for SparseTensor<T>
where
    T: Number,
{
    fn rank(&self) -> usize {
        self.sizes.len()
    }

    fn size(&self, rank: RankIndex) -> usize {
        self.sizes[rank]
    }

    fn elem(&self, indices: &[usize]) -> T {
        self[indices]
    }

    fn elem_mut(&mut self, indices: &[usize]) -> &mut T {
        &mut self[indices]
    }
}

impl<T> From<Vec<T>> for SparseTensor<T>
where
    T: Number,
{
    fn from(vec: Vec<T>) -> Self {
        let sizes = vec![vec.len()];
        let elems = vec
            .into_iter()
            .enumerate()
            .map(|(i, v)| (vec![i], v))
            .collect();

        Self {
            sizes,
            elems,
            default: T::default(),
        }
    }
}
