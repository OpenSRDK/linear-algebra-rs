use crate::number::Number;
use crate::{ge::Matrix, matrix::*};
use rayon::prelude::*;

pub mod trf;
pub mod trs;

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
    pub fn new(dim: usize) -> Self {
        let e = vec![T::default(); dim.max(1) - 1];
        Self {
            dl: e.clone(),
            d: vec![T::default(); dim],
            du: e,
        }
    }

    /// - `dl`: Lower diagonal elements. The length must be `dimension - 1`.
    /// - `d`: Diagonal elements. The length must be `dimension`.
    /// - `du`: Upper diagonal elements. The length must be `dimension - 1`.
    pub fn from(dl: Vec<T>, d: Vec<T>, du: Vec<T>) -> Result<Self, MatrixError> {
        let n_1 = d.len().max(1) - 1;
        if n_1 != dl.len() || n_1 != du.len() {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { dl, d, du })
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.d.len()
    }

    /// Lower diagonal elements.
    pub fn dl(&self) -> &[T] {
        &self.dl
    }

    /// Diagonal elements.
    pub fn d(&self) -> &[T] {
        &self.d
    }

    /// Lower diagonal elements.
    pub fn du(&self) -> &[T] {
        &self.du
    }

    /// Returns `(self.dl, self.d, self.du)`
    pub fn eject(self) -> (Vec<T>, Vec<T>, Vec<T>) {
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

        mat.elems_mut()
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
