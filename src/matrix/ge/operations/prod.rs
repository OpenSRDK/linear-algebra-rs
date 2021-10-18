use crate::{matrix::Matrix, number::Number};
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Linear product
    pub fn linear_prod(&self, rhs: &Matrix<T>) -> T {
        if !self.same_size(rhs) {
            panic!("Dimension mismatch.")
        } else {
            self.elems
                .par_iter()
                .zip(rhs.elems.par_iter())
                .map(|(&s, &r)| s * r)
                .sum()
        }
    }

    /// # Hamadard product
    pub fn hadamard_prod(self, rhs: &Matrix<T>) -> Matrix<T> {
        if !self.same_size(rhs) {
            panic!("Dimension mismatch.")
        }
        let mut slf = self;

        slf.elems
            .par_iter_mut()
            .zip(rhs.elems.par_iter())
            .map(|(s, &r)| {
                *s *= r;
            })
            .collect::<Vec<_>>();

        slf
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = DiagonalMatrix::<f64>::identity(2).mat();
        let b = DiagonalMatrix::<f64>::identity(2).mat();
        let c = a.hadamard_prod(&b);

        assert_eq!(b[0][0], c[0][0])
    }
}
