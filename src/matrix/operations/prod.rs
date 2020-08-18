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
                .map(|(&s, &r)| s * r)
                .sum()
        }
    }

    /// # Hamadard product
    pub fn hadamard_prod<V: Type>(mut self, rhs: &Matrix<V, U>) -> Matrix<Standard, U> {
        if !self.is_same_size(rhs) {
            panic!("dimension mismatch")
        }

        self.elements
            .par_iter_mut()
            .zip(rhs.elements.par_iter())
            .map(|(s, r)| {
                *s *= *r;
            })
            .collect::<Vec<_>>();

        self.transmute()
    }

    /// # Kronecker product
    pub fn kronecker_prod<V: Type>(&self, rhs: &Matrix<V, U>) -> Matrix<Standard, U> {
        let sn = self.rows;
        let sm = self.columns;
        let rn = rhs.rows;
        let rm = rhs.columns;
        let n = sn * rn;
        let m = sm * rm;
        let mut matrix = Matrix::<Standard, U>::zeros(n, m);

        for si in 0..sn {
            for sj in 0..sm {
                for ri in 0..rn {
                    for rj in 0..rm {
                        matrix[si * sn + ri][sj * sn + rj] = self[si][sj] * rhs[ri][rj];
                    }
                }
            }
        }

        matrix
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
