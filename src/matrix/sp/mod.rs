use crate::number::Number;
use std::collections::HashMap;

pub mod mul;

#[derive(Debug)]
pub struct SparseMatrix<T = f64>
where
    T: Number,
{
    rows: usize,
    cols: usize,
    elems: HashMap<(usize, usize), T>,
}

impl<T> SparseMatrix<T>
where
    T: Number,
{
    pub fn new(rows: usize, cols: usize, elems: HashMap<(usize, usize), T>) -> Self {
        Self { rows, cols, elems }
    }

    pub fn t(&self) -> Self {
        Self::new(
            self.cols,
            self.rows,
            self.elems
                .iter()
                .map(|(&(i, j), &value)| ((j, i), value))
                .collect::<HashMap<(usize, usize), T>>(),
        )
    }
}
