use std::ops::{Index, IndexMut};

use crate::{sparse::SparseTensor, Number, TensorError};

impl<T> Index<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() != self.sizes.len() {
            panic!("{}", TensorError::RankMismatch);
        }
        for (rank, &d) in index.iter().enumerate() {
            if self.sizes[rank] <= d {
                panic!("{}", TensorError::OutOfRange);
            }
        }

        self.elems.get(index).unwrap_or(&self.default)
    }
}

impl<T> IndexMut<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if index.len() != self.sizes.len() {
            panic!("{}", TensorError::RankMismatch);
        }
        for (rank, &d) in index.iter().enumerate() {
            if self.sizes[rank] <= d {
                panic!("{}", TensorError::OutOfRange);
            }
        }

        self.elems.entry(index.to_vec()).or_default()
    }
}
