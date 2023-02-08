use crate::{Matrix, Number, Tensor};

impl<T> Tensor<T> for Matrix<T>
where
    T: Number,
{
    fn levels(&self) -> usize {
        2
    }

    fn dim(&self, level: usize) -> usize {
        match level {
            0 => self.rows(),
            1 => self.cols(),
            _ => 0,
        }
    }

    fn elem(&self, indices: &[usize]) -> T {
        self[(indices[0], indices[1])]
    }

    fn elem_mut(&mut self, indices: &[usize]) -> &mut T {
        &mut self[(indices[0], indices[1])]
    }
}
