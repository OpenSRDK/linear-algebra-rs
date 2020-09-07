use crate::number::Number;

#[derive(Clone, Debug, Default, Hash)]
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

    pub fn elems(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }
}
