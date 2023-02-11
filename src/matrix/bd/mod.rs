use crate::{number::Number, Matrix, MatrixError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Hash, Serialize, Deserialize)]
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
    pub fn new(dim: usize) -> Self {
        Self {
            d: vec![T::default(); dim],
            e: vec![T::default(); dim.max(1) - 1],
        }
    }

    /// - `d`: Diagonal elements. The length must be `dimension`.
    /// - `e`: First superdiagonal or subdiagonal elements. The length must be `dimension - 1`.
    pub fn from(d: Vec<T>, e: Vec<T>) -> Result<Self, MatrixError> {
        if d.len().max(1) - 1 != e.len() {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { d, e })
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.d.len()
    }

    /// Diagonal elements.
    pub fn d(&self) -> &[T] {
        &self.d
    }

    /// first superdiagonal or subdiagonal elements.
    pub fn e(&self) -> &[T] {
        &self.e
    }

    /// Returns `(self.d, self.e)`
    pub fn eject(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn mat(&self, upper: bool) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::new(n, n);

        // for i in 0..n {
        //   mat[i][i] = self.d[i]
        // }

        // if upper {
        //     for i in 0..n - 1 {
        //         mat[i + 1][i] = self.e[i];
        //     }
        // } else {
        //     for i in 0..n - 1 {
        //         mat[i][i + 1] = self.e[i];
        //     }
        // }

        mat.elems_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(k, elem)| ((k / n, k % n), elem))
            .for_each(|((i, j), elem)| {
                if i == j {
                    *elem = self.d[i];
                } else if i + 1 == j && upper {
                    *elem = self.e[i];
                } else if i == j + 1 && !upper {
                    *elem = self.e[j];
                }
            });

        mat
    }
}
