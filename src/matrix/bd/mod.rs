use std::error::Error;

use crate::{matrix::*, number::Number};

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
    pub fn new(d: Vec<T>, e: Vec<T>) -> Result<Self, Box<dyn Error>> {
        if d.len().max(1) - 1 != e.len() {
            return Err(MatrixError::DimensionMismatch.into());
        }

        Ok(Self { d, e })
    }

    pub fn d(&self) -> &[T] {
        &self.d
    }

    pub fn e(&self) -> &[T] {
        &self.e
    }

    pub fn elems(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn mat(&self, upper: bool) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::diag(&self.d);

        if upper {
            for i in 0..n - 1 {
                mat[i + 1][i] = self.e[i];
            }
        } else {
            for i in 0..n - 1 {
                mat[i][i + 1] = self.e[i];
            }
        }

        mat
    }
}
