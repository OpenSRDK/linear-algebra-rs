use crate::number::*;
use crate::Matrix;
use crate::MatrixError;

pub mod pp;

pub mod trf;
pub mod tri;
pub mod trs;

#[derive(Clone, Debug, Default, Hash)]
pub struct SymmetricPackedMatrix<T = f64>
where
    T: Number,
{
    dim: usize,
    elems: Vec<T>,
}

impl<T> SymmetricPackedMatrix<T>
where
    T: Number,
{
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            elems: vec![T::default(); dim * (dim + 1) / 2],
        }
    }

    /// You can do `unwrap()` if you have a conviction that `elems.len() == dim * (dim + 1) / 2`
    pub fn from(dim: usize, elems: Vec<T>) -> Result<Self, MatrixError> {
        if elems.len() != dim * (dim + 1) / 2 {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { dim, elems })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn eject(self) -> Vec<T> {
        self.elems
    }

    pub fn elems(&self) -> &[T] {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut [T] {
        &mut self.elems
    }

    pub fn from_mat(mat: &Matrix<T>) -> Result<Self, MatrixError> {
        let n = mat.rows();
        if n != mat.cols() {
            return Err(MatrixError::DimensionMismatch);
        }

        let elems = (0..n)
            .into_iter()
            .map(|j| (j, &mat[j]))
            .flat_map(|(j, col)| col[j..n].into_iter())
            .map(|e| *e)
            .collect::<Vec<_>>();
        Self::from(n, elems)
    }

    pub fn to_mat(&self) -> Matrix<T> {
        let n = self.dim;
        let elems = (0..n)
            .into_iter()
            .flat_map(|j| {
                let index = n * (n + 1) / 2 - (n - j) * (n - j + 1) / 2;
                vec![T::default(); j]
                    .into_iter()
                    .chain(self.elems[index..index + (n - j)].iter().map(|e| *e))
            })
            .collect::<Vec<_>>();

        Matrix::<T>::from(n, elems).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_works() {
        let a = mat!(
           1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
           2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
           4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
           7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
          11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
          16.0, 17.0, 18.0, 19.0, 20.0, 21.0
        );

        let ap = SymmetricPackedMatrix::from_mat(&a).unwrap();
        let n = ap.dim();

        assert_eq!(ap.elems()[n * (n + 1) / 2 - 1], 21.0);

        let a2 = ap.to_mat();

        assert_eq!(a, a2);
    }
}
