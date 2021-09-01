use crate::{DiagonalMatrix, Number};
use std::ops::{Index, IndexMut};

impl<T> Index<usize> for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.d[index]
    }
}

impl<T> IndexMut<usize> for DiagonalMatrix<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d[index]
    }
}
