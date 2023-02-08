pub mod matrix;
pub mod sparse;

use std::error::Error;

use crate::Number;

pub trait Tensor<T>
where
    T: Number,
{
    fn levels(&self) -> usize;
    fn dim(&self, level: usize) -> usize;
    fn elem(&self, indices: &[usize]) -> T;
    fn elem_mut(&mut self, indices: &[usize]) -> &mut T;
}

#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}
