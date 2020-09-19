use crate::matrix::MatrixError;
use crate::{matrix::*, number::Number};
use std::error::Error;

#[derive(Clone, Debug, Default, Hash)]
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
        let bigp = self.k.len();

        let mut row_block = 1;
        let mut col_block = 1;

        for p in (0..bigp).rev() {
            for j in 0..self.cols {
                for i in 0..self.rows {
                    new_matrix[j][i] *=
                        self.k[p][j / col_block % self.k[p].cols][i / row_block % self.k[p].rows];
                }
            }

            row_block *= self.k[p].rows;
            col_block *= self.k[p].cols;
        }

        new_matrix
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: Vec<f64>) -> Result<Vec<f64>, Box<dyn Error>> {
        let n = v.len();

        if self.cols != n {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let bigp = self.k.len();
        let mut u = v.col_mat();

        for p in (0..bigp).rev() {
            let bigu_rows = self.k[p].cols;
            let bigu = u.reshape(bigu_rows);
            let k_bigu = &self.k[p] * bigu;

            u = k_bigu.t().vec().col_mat();
        }

        Ok(u.vec())
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

        println!("c {:#?}", c);

        assert_eq!(c[(0, 0)], 1.0);
        assert_eq!(c[(0, 3)], 4.0);
        assert_eq!(c[(2, 1)], 6.0);

        let ab1 = ab.vec_mul(vec![1.0; 4]).unwrap().col_mat();
        let c1 = &c * vec![1.0; 4].col_mat();

        assert_eq!(ab1[(0, 0)], c1[(0, 0)]);
        assert_eq!(ab1[(1, 0)], c1[(1, 0)]);
    }
}
