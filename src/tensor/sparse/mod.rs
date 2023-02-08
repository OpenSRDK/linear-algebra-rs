pub mod operations;
pub mod operators;

use crate::{Number, Tensor};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct SparseTensor<T = f64>
where
    T: Number,
{
    dims: Vec<usize>,
    elems: HashMap<Vec<usize>, T>,
    default: T,
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn new(dims: Vec<usize>) -> Self {
        Self {
            dims,
            elems: HashMap::new(),
            default: T::default(),
        }
    }

    pub fn from(dims: Vec<usize>, elems: HashMap<Vec<usize>, T>) -> Self {
        Self {
            dims,
            elems,
            default: T::default(),
        }
    }

    pub fn same_size(&self, rhs: &SparseTensor<T>) -> bool {
        self.dims == rhs.dims
    }
}

impl<T> Tensor<T> for SparseTensor<T>
where
    T: Number,
{
    fn levels(&self) -> usize {
        self.dims.len()
    }

    fn dim(&self, level: usize) -> usize {
        self.dims[level]
    }

    fn elem(&self, indices: &[usize]) -> T {
        self[indices]
    }

    fn elem_mut(&mut self, indices: &[usize]) -> &mut T {
        &mut self[indices]
    }
}
