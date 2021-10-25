use crate::{number::Number, Matrix};
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Determinant
    /// for triangle matrix
    /// To apply this method to none triangle matrix, use LU decomposition or Cholesky decomposition.
    pub fn trdet(&self) -> T {
        (0..self.rows)
            .into_par_iter()
            .map(|i| self[(i, i)])
            .product()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 0.0;
            3.0, 4.0
        );
        let det = a.trdet();
        assert_eq!(det, 4.0);
    }
}
