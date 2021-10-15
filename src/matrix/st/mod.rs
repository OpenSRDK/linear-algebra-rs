use crate::matrix::*;
use crate::number::Number;
use rayon::prelude::*;

pub mod ev;
pub mod evd;

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

    pub fn mat(&self) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::new(n, n);

        // for i in 0..n {
        //     mat[i][i] = self.d[i];
        // }

        // for i in 0..n - 1 {
        //     mat[i][i + 1] = self.e[i];
        //     mat[i + 1][i] = self.e[i];
        // }

        (0..n).into_par_iter().for_each(|i| {
            mat[i][i] = self.d[i];
        });

        (0..n - 1).into_par_iter().for_each(|i| {
            mat[i][i + 1] = self.e[i];
            mat[i + 1][i] = self.e[i];
        });

        mat
    }
}
