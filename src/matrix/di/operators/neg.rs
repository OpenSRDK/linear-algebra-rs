use super::mul::mul_scalar;
use crate::{DiagonalMatrix, Number};
use std::ops::Neg;

impl<T> Neg for DiagonalMatrix<T>
where
    T: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        mul_scalar(-T::one(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = -DiagonalMatrix::from(vec![1.0, 2.0]);
        assert_eq!(a[0], -1.0);
    }
}
