
use crate::{matrix::Matrix, number::Number, types::Type};

pub trait SubMatrix<U>: Send + Sync {
    fn size(&self) -> (usize, usize);
    fn index(&self, i: usize, j: usize) -> U;
}

impl<U> SubMatrix<U> for U
where
    U: Number,
{
    fn size(&self) -> (usize, usize) {
        (1usize, 1usize)
    }

    fn index(&self, _: usize, _: usize) -> U {
        *self
    }
}

impl<'a, U> SubMatrix<U> for &'a [U]
where
    U: Number,
{
    fn size(&self) -> (usize, usize) {
        (1usize, self.len())
    }

    fn index(&self, _: usize, j: usize) -> U {
        self[j]
    }
}

impl<T, U> SubMatrix<U> for Matrix<T, U>
where
    T: Type,
    U: Number,
{
    fn size(&self) -> (usize, usize) {
        (self.get_rows(), self.get_columns())
    }

    fn index(&self, i: usize, j: usize) -> U {
        self[i][j]
    }
}

impl<T, U> SubMatrix<U> for &Matrix<T, U>
where
    T: Type,
    U: Number,
{
    fn size(&self) -> (usize, usize) {
        (self.get_rows(), self.get_columns())
    }

    fn index(&self, i: usize, j: usize) -> U {
        self[i][j]
    }
}
