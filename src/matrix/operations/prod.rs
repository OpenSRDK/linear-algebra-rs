use crate::{
    matrix::Matrix,
    number::Number,
    types::{Standard, Type},
};
use rayon::prelude::*;

impl<T, U> Matrix<T, U>
where
    T: Type,
    U: Number,
{
    /// # Linear product
    pub fn linear_prod<V: Type>(&self, rhs: &Matrix<V, U>) -> U {
        if !self.is_same_size(rhs) {
            panic!("dimension mismatch")
        } else {
            self.elements
                .par_iter()
                .zip(rhs.elements.par_iter())
                .map(|(s, r)| *s * *r)
                .sum()
        }
    }

    /// # Hamadard product
    pub fn hadamard_prod<V: Type>(self, rhs: &Matrix<V, U>) -> Matrix<Standard, U> {
        if !self.is_same_size(rhs) {
            panic!("dimension mismatch")
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = identity::<f64>(2);
        let b = identity::<f64>(2);
        let c = a.hadamard_prod(&b);

        assert_eq!(b[0][0], c[0][0])
    }
}
