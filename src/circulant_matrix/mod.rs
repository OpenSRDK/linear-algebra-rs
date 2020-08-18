use crate::number::Number;

pub mod real;

pub struct CirculantMatrix<U>
where
    U: Number,
{
    row: Vec<U>,
}

impl<U> CirculantMatrix<U>
where
    U: Number,
{
    pub fn new(row: Vec<U>) -> Self {
        Self { row }
    }
}
