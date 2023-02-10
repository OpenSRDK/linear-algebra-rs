pub mod operations;
pub mod operators;

use crate::{Matrix, Number, Tensor};
use std::collections::HashMap;

pub type RankIndex = usize;

#[derive(Clone, Debug, PartialEq)]
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
        if sizes.iter().product::<usize>() == 0 {
            panic!("SparseTensor::new() is not available for zero-sized tensor.");
        }
        Self {
            sizes,
            elems: HashMap::new(),
            default: T::default(),
        }
    }

    pub fn from(sizes: Vec<usize>, elems: HashMap<Vec<usize>, T>) -> Self {
        if sizes.iter().product::<usize>() == 0 {
            panic!("SparseTensor::from() is not available for zero-sized tensor.");
        }
        Self {
            sizes,
            elems,
            default: T::default(),
        }
    }

    pub fn is_same_size(&self, rhs: &SparseTensor<T>) -> bool {
        self.sizes == rhs.sizes
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
