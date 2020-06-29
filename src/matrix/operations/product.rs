use crate::{types::{Standard, Type}, number::Number, matrix::Matrix};
use rayon::prelude::*;

impl<T, U> Matrix<T, U>
where
    T: Type,
    U: Number,
{
    pub fn linear_product<V: Type>(&self, rhs: &Matrix<V, U>) -> U {
        if !self.is_same_size(rhs) {
            panic!("different dimensions")
        } else {
            self.elements
                .par_iter()
                .zip(rhs.elements.par_iter())
                .map(|(s, r)| *s * *r)
                .sum()
        }
    }

    pub fn hadamard_product<V: Type>(self, rhs: &Matrix<V, U>) -> Matrix<Standard, U> {
        if !self.is_same_size(rhs) {
            panic!("different dimensions")
        }
        let mut slf = self;

        slf.elements
            .par_iter_mut()
            .zip(rhs.elements.par_iter())
            .map(|(s, r)| {
                *s *= *r;
            })
            .collect::<Vec<_>>();

        slf.transmute()
    }
}
