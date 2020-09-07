use std::error::Error;

use crate::matrix::*;
use crate::number::Number;

#[derive(Clone, Debug, Default, Hash)]
pub struct SymmetricTridiagonalMatrix<T = f64>
where
    T: Number,
{
    d: Vec<T>,
    e: Vec<T>,
}

impl<T> SymmetricTridiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(d: Vec<T>, e: Vec<T>) -> Result<Self, Box<dyn Error>> {
        if d.len().min(1) - 1 != e.len() {
            return Err(MatrixError::DimensionMismatch.into());
        }

        Ok(Self { d, e })
    }

    pub fn elems(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn mat(&self) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::diag(&self.d);

        for i in 0..n - 1 {
            mat[i][i + 1] = self.e[i];
            mat[i + 1][i] = self.e[i];
        }

        mat
    }
}
