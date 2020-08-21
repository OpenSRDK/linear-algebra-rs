use crate::matrix::Matrix;
use crate::number::Number;
use std::ops::{Index, IndexMut};

impl<T> Index<usize> for Matrix<T>
where
    T: Number,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        let i = self.columns * index;
        &self.elements[i..i + self.columns]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = self.columns * index;
        &mut self.elements[i..i + self.columns]
    }
}
