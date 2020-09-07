pub mod gv;

use crate::number::Number;

#[derive(Clone, Debug, Default, Hash)]
pub struct CirculantMatrix<T = f64>
where
    T: Number,
{
    row_elems: Vec<T>,
}

impl<T> CirculantMatrix<T>
where
    T: Number,
{
    pub fn new(row_elems: Vec<T>) -> Self {
        Self { row_elems }
    }

    pub fn row_elems(&self) -> &[T] {
        &self.row_elems
    }
}
