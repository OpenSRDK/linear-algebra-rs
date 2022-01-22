use crate::{ge::Matrix, Number};
use rayon::prelude::*;

pub mod operators;
pub mod powf;
pub mod powi;

#[derive(Clone, Debug, Default, PartialEq, Hash)]
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
    /// - `d`: Diagonal elements. The length must be `dimension`.
    pub fn new(d: Vec<T>) -> Self {
        Self { d }
    }

    /// Creates an identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::new(vec![T::one(); n])
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.d.len()
    }

    /// Diagonal elements.
    pub fn d(&self) -> &[T] {
        &self.d
    }

    /// Returns `self.d`
    pub fn eject(self) -> Vec<T> {
        self.d
    }

    pub fn mat(&self) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::<T>::new(n, n);

        // for i in 0..n {
        //     mat[i][i] = self.d[i];
        // }

        mat.elems_mut()
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
