use crate::matrix::ge::Matrix;
use crate::number::Number;
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Transpose
    pub fn t(&self) -> Matrix<T> {
        let elems = (0..self.rows)
            .into_par_iter()
            .flat_map(|i| (0..self.cols).into_par_iter().map(move |j| (i, j)))
            .map(|(i, j)| self[(i, j)])
            .collect();

        Matrix::from(self.cols, elems).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            1.0, 3.0;
            2.0, 4.0;
            3.0, 6.0
        ];
        let at = a.t();

        assert_eq!(at[(1, 0)], 3.0);
        assert_eq!(at[(1, 2)], 6.0);
    }
}
