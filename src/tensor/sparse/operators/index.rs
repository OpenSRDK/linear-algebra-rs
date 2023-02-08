use std::ops::{Index, IndexMut};

use crate::{sparse::SparseTensor, Number};

impl<T> Index<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.elems.get(index).unwrap_or(&self.default)
    }
}

impl<T> IndexMut<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.elems.entry(index.to_vec()).or_default()
    }
}
