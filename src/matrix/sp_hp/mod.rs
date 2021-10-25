use crate::number::*;
use crate::MatrixError;

pub mod pp;

pub mod trf;
pub mod tri;
pub mod trs;

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
}
