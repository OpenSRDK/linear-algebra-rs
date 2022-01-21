use crate::{Matrix, Number};
use std::iter::Sum;

impl<T> Sum<Matrix<T>> for Matrix<T>
where
    T: Number,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = [mat!(
            1.0, 2.0;
            3.0, 4.0
        ); 3]
            .into_iter()
            .sum::<Matrix>();

        assert_eq!(a[(0, 0)], 3.0);
    }
}
