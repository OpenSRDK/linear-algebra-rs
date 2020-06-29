use crate::types::Type;
use crate::matrix::Matrix;
use crate::number::{Number};
use std::ops::{Index, IndexMut};

impl<T, U> Index<usize> for Matrix<T, U>
where
    T: Type,
    U: Number,
{
    type Output = [U];
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.rows {
            panic!()
        }
        let i = self.columns * index;
        &self.elements[i..i + self.columns]
    }
}

impl<T, U> IndexMut<usize> for Matrix<T, U>
where
    T: Type,
    U: Number,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.columns {
            panic!()
        }
        let i = self.columns * index;
        &mut self.elements[i..i + self.columns]
    }
}
