pub mod gv;

use crate::number::Number;

pub struct CirculantMatrix<T = f64>
where
    T: Number,
{
    row: Vec<T>,
}

impl<T> CirculantMatrix<T>
where
    T: Number,
{
    pub fn new(row: Vec<T>) -> Self {
        Self { row }
    }
}
