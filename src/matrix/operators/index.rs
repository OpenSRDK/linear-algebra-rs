use crate::matrix::Matrix;
use crate::number::Number;
use std::ops::{Index, IndexMut};

impl<T> Index<usize> for Matrix<T>
where
    T: Number,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        let i = self.rows * index;
        &self.elems[i..i + self.rows]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = self.rows * index;
        &mut self.elems[i..i + self.rows]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Number,
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.elems[index.0 + index.1 * self.rows]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.elems[index.0 + index.1 * self.rows]
    }
}
