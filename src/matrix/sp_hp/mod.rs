use crate::number::Number;
use crate::MatrixError;

pub mod trf;
pub mod trs;

pub mod pp;

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

    pub fn from(dim: usize, elems: Vec<T>) -> Result<Self, MatrixError> {
        if elems.len() != dim * (dim + 1) / 2 {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { dim, elems })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vec(self) -> Vec<T> {
        self.elems
    }

    pub fn slice(&self) -> &[T] {
        &self.elems
    }
}
