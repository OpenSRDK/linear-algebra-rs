use crate::{number::Number, macros::sub_matrix::SubMatrix};

#[macro_export]
macro_rules! zeros {
    ($($e: expr),+) => {
        {
            use $crate::macros::zeros::Zeros;
            Zeros($($e),+)
        }
    };
}

pub struct Zeros(pub usize, pub usize);

impl<U: Number> SubMatrix<U> for Zeros {
    fn size(&self) -> (usize, usize) {
        (self.0, self.1)
    }

    fn index(&self, _: usize, _: usize) -> U {
        U::default()
    }
}
