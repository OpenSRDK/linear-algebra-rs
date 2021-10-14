use crate::matrix::Matrix;
use crate::number::c64;
use rayon::prelude::*;

impl Matrix<c64> {
    pub fn adjoint(&self) -> Matrix<c64> {
        let elems = (0..self.rows)
            .into_par_iter()
            .flat_map(|i| (0..self.cols).into_par_iter().map(move |j| (i, j)))
            .map(|(i, j)| self[(i, j)].conj())
            .collect();

        Matrix::<c64>::from(self.cols, elems)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let mut a = Matrix::<c64>::new(2, 3);
        a[(1, 2)] = c64::new(2.0, 3.0);
        let b = a.adjoint();

        assert_eq!(b[(2, 1)], c64::new(2.0, -3.0))
    }
}
