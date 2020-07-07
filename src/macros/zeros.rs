use crate::{macros::sub_matrix::SubMatrix, number::Number};

#[macro_export]
macro_rules! zeros {
    ($e1: expr, $e2: expr) => {{
        use $crate::macros::zeros::Zeros;
        Zeros($e1, $e2)
    }};
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 0.0;
            0.0, 1.0
        );
        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[0][1], 0.0);
        assert_eq!(a[1][0], 0.0);
        assert_eq!(a[1][1], 1.0);

        let b = mat!(
            &a, zeros!(2, 2);
            zeros!(2, 2), &a
        );

        assert_eq!(b[0][0], 1.0);
        assert_eq!(b[0][1], 0.0);
        assert_eq!(b[3][0], 0.0);
        assert_eq!(b[3][3], 1.0);
    }
}
