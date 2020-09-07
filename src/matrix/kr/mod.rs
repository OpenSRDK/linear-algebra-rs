use crate::matrix::MatrixError;
use crate::{matrix::*, number::Number};
use std::error::Error;

#[derive(Clone, Debug)]
pub struct KroneckerMatrices<T = f64>
where
    T: Number,
{
    k: Vec<Matrix<T>>,
    rows: usize,
    cols: usize,
}

impl<T> KroneckerMatrices<T>
where
    T: Number,
{
    pub fn new(k: Vec<Matrix<T>>) -> Self {
        let (rows, cols) = k
            .iter()
            .fold((1usize, 1usize), |v, m| (v.0 * m.rows, v.1 * m.cols));
        Self { k, rows, cols }
    }

    pub fn elems_ref(&self) -> &[Matrix<T>] {
        &self.k
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn prod(&self) -> Matrix<T> {
        let mut new_matrix = Matrix::from(self.rows, vec![T::one(); self.rows * self.cols]);
        let k_len = self.k.len();

        let mut row_block = 1;
        let mut col_block = 1;

        for p in (0..k_len).rev() {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    new_matrix[i][j] *=
                        self.k[p][i / row_block % self.k[p].rows][j / col_block % self.k[p].cols];
                }
            }

            row_block *= self.k[p].rows;
            col_block *= self.k[p].cols;
        }

        new_matrix
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        if self.k.len() == 0 || self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::Empty.into());
        }

        let n = v.len();

        if self.cols != n {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let k_len = self.k.len();
        let mut u = v.to_vec().col_mat();
        for p in (0..k_len).rev() {
            let bigu_rows = self.k[p].cols;
            let bigu_cols = u.rows() / bigu_rows;
            // reshape(u, bigu_rows, bigu_cols)
            let bigut = Matrix::from(bigu_cols, u.elems);
            let bigut_kt = bigut * self.k[p].t();

            u = bigut_kt.elems().col_mat();
        }

        Ok(u.elems())
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let b = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let ab = KroneckerMatrices::new(vec![a, b]);
        let c = ab.prod();

        assert_eq!(c[0][0], 1.0);
        assert_eq!(c[0][3], 4.0);
        assert_eq!(c[2][1], 6.0);

        let ab1 = ab.vec_mul(&[1.0; 4]).unwrap().col_mat();
        let c1 = &c * vec![1.0; 4].col_mat();

        assert_eq!(ab1[0][0], c1[0][0]);
        assert_eq!(ab1[1][0], c1[1][0]);
    }
}
