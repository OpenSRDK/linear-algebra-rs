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
    /// `d`: diagonal elements
    /// `e`: first superdiagonal or subdiagonal elements
    pub fn new(d: Vec<T>, e: Vec<T>) -> Result<Self, MatrixError> {
        if d.len().max(1) - 1 != e.len() {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { d, e })
    }

    pub fn n(&self) -> usize {
        self.d.len()
    }

    /// diagonal elements
    pub fn d(&self) -> &[T] {
        &self.d
    }

    /// first superdiagonal or subdiagonal elements
    pub fn e(&self) -> &[T] {
        &self.e
    }

    pub fn eject(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn mat(&self, upper: bool) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::new(n, n);

        mat.elems
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
