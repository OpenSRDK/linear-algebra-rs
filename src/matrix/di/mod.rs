use crate::{Matrix, Number};

pub mod inv;
pub mod operators;
pub mod powf;
pub mod powi;

#[derive(Clone, Debug, Default, Hash)]
pub struct DiagonalMatrix<T = f64>
where
    T: Number,
{
    d: Vec<T>,
}

impl<T> DiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(n: usize) -> Self {
        Self::from(vec![T::default(); n])
    }

    pub fn from(d: Vec<T>) -> Self {
        Self { d }
    }

    pub fn identity(n: usize) -> Self {
        Self::from(vec![T::one(); n])
    }

    pub fn n(&self) -> usize {
        self.d.len()
    }

    pub fn d(&self) -> &[T] {
        &self.d
    }

    pub fn eject(self) -> Vec<T> {
        self.d
    }

    pub fn mat(&self) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::<T>::new(n, n);
        for i in 0..n {
            mat[i][i] = self.d[i];
        }

        mat
    }
}
