use super::CirculantMatrix;
use crate::number::Number;
pub struct ToeplitzMatrix<U>
where
    U: Number,
{
    dim: usize,
    row: Vec<U>,
    column: Vec<U>,
}

impl<U> ToeplitzMatrix<U>
where
    U: Number,
{
    pub fn new(row: Vec<U>, column: Vec<U>) -> Self {
        let dim = row.len();

        if column.len() != dim {
            panic!("different dimensions")
        }
        if dim < 2 || row[0] != column[0] {
            panic!("")
        }

        Self { dim, row, column }
    }

    pub fn embedded_circulant(&self) -> CirculantMatrix<U> {
        let row = (0..self.dim)
            .into_iter()
            .chain((1..self.dim - 1).rev().into_iter())
            .map(|i| self.row[i])
            .collect();

        CirculantMatrix::<U>::new(row)
    }
}
