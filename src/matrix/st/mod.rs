use crate::number::Number;

pub struct SymmetricTridiagonalMatrix<T = f64>
where
    T: Number,
{
    d: Vec<T>,
    e: Vec<T>,
}

impl<T> SymmetricTridiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(d: Vec<T>, e: Vec<T>) -> Self {
        Self { d, e }
    }

    pub fn get_elements(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }
}
