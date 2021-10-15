use crate::{matrix::*, number::Number};
use rayon::prelude::*;

#[derive(Clone, Debug, Default, Hash)]
pub struct BidiagonalMatrix<T = f64>
where
    T: Number,
{
    d: Vec<T>,
    e: Vec<T>,
}

impl<T> BidiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(d: Vec<T>, e: Vec<T>) -> Result<Self, MatrixError> {
        if d.len().max(1) - 1 != e.len() {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { d, e })
    }

    pub fn n(&self) -> usize {
        self.d.len()
    }

    pub fn d(&self) -> &[T] {
        &self.d
    }

    pub fn e(&self) -> &[T] {
        &self.e
    }

    pub fn eject(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn mat(&self, upper: bool) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::new(n, n);
        (0..n).into_par_iter().for_each(|i| mat[i][i] = self.d[i]);

        if upper {
            (0..n - 1)
                .into_par_iter()
                .for_each(|i| mat[i + 1][i] = self.e[i]);
        } else {
            (0..n - 1)
                .into_par_iter()
                .for_each(|i| mat[i][i + 1] = self.e[i]);
        }

        mat
    }
}
