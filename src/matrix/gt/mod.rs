pub mod trf;
pub mod trs;

use crate::matrix::*;
use crate::number::Number;

#[derive(Clone, Debug, Default, Hash)]
pub struct TridiagonalMatrix<T = f64>
where
    T: Number,
{
    dl: Vec<T>,
    d: Vec<T>,
    du: Vec<T>,
}

impl<T> TridiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(dl: Vec<T>, d: Vec<T>, du: Vec<T>) -> Result<Self, MatrixError> {
        let n_1 = d.len().max(1) - 1;
        if n_1 != dl.len() || n_1 != du.len() {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { dl, d, du })
    }

    pub fn dl(&self) -> &[T] {
        &self.dl
    }

    pub fn d(&self) -> &[T] {
        &self.d
    }

    pub fn du(&self) -> &[T] {
        &self.du
    }

    pub fn elems(self) -> (Vec<T>, Vec<T>, Vec<T>) {
        (self.dl, self.d, self.du)
    }

    pub fn mat(&self) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::new(n, n);

        // for i in 0..n {
        //     mat[i][i] = self.d[i];
        // }

        // for i in 0..n - 1 {
        //     mat[i][i + 1] = self.du[i];
        //     mat[i + 1][i] = self.dl[i];
        // }

        mat.elems
            .par_iter_mut()
            .enumerate()
            .map(|(k, elem)| ((k / n, k % n), elem))
            .for_each(|((i, j), elem)| {
                if i == j {
                    *elem = self.d[i];
                } else if i + 1 == j {
                    *elem = self.du[i];
                } else if i == j + 1 {
                    *elem = self.dl[j];
                }
            });
 
        mat
    }
}
