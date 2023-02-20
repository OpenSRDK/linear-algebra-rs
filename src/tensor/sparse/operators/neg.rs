use crate::{sparse::SparseTensor, Number};
use std::ops::Neg;

impl<T> Neg for SparseTensor<T>
where
    T: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * (-T::one())
    }
}

#[cfg(test)]
mod tests {
    use crate::{sparse::SparseTensor, *};
    #[test]
    fn it_works() {
        let mut a = SparseTensor::new(vec![3, 2, 2]);
        a[&[0, 0, 0]] = 2.0;
        a[&[0, 0, 1]] = 4.0;
        a[&[1, 1, 0]] = 2.0;
        a[&[1, 1, 1]] = 4.0;
        a[&[2, 0, 0]] = 2.0;
        a[&[2, 0, 1]] = 4.0;

        let a = -a;

        let mut b = SparseTensor::new(vec![3, 2, 2]);
        b[&[0, 0, 0]] = -2.0;
        b[&[0, 0, 1]] = -4.0;
        b[&[1, 1, 0]] = -2.0;
        b[&[1, 1, 1]] = -4.0;
        b[&[2, 0, 0]] = -2.0;
        b[&[2, 0, 1]] = -4.0;

        assert_eq!(a[&[0, 0, 0]], b[&[0, 0, 0]]);
        assert_eq!(a[&[0, 0, 1]], b[&[0, 0, 1]]);
        assert_eq!(a[&[1, 1, 0]], b[&[1, 1, 0]]);
        assert_eq!(a[&[1, 1, 1]], b[&[1, 1, 1]]);
        assert_eq!(a[&[2, 0, 0]], b[&[2, 0, 0]]);
        assert_eq!(a[&[2, 0, 1]], b[&[2, 0, 1]]);
    }
}
