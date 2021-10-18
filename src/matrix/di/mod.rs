use crate::{Matrix, Number};
use rayon::prelude::*;

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
    pub fn new(d: Vec<T>) -> Self {
        Self { d }
    }

    pub fn identity(n: usize) -> Self {
        Self::new(vec![T::one(); n])
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
        // for i in 0..n {
        //     mat[i][i] = self.d[i];
        // }
        mat.elems
            .par_iter_mut()
            .enumerate()
            .map(|(k, elem)| ((k / n, k % n), elem))
            .for_each(|((i, j), elem)| {
              if i == j {
                *elem = self.d[i];
              }
            });

        mat
    }
}

pub trait VectorDiag<T>
where
    T: Number,
{
    fn diag(self) -> DiagonalMatrix<T>;
}

impl<T> VectorDiag<T> for Vec<T>
where
    T: Number,
{
    fn diag(self) -> DiagonalMatrix<T> {
        DiagonalMatrix::<T>::new(self)
    }
}
