use super::SparseMatrix;
use crate::number::Number;
use std::ops::Index;
use std::ops::IndexMut;

impl<T> Index<(usize, usize)> for SparseMatrix<T>
where
    T: Number,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.elems.get(&index).unwrap_or(&self.default)
    }
}

impl<T> IndexMut<(usize, usize)> for SparseMatrix<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.elems.entry(index).or_default()
    }
}
