use crate::matrix::ge::*;
use crate::matrix::MatrixError;
use crate::number::Number;

#[derive(Clone, Debug, Default, PartialEq, Hash)]
pub struct KroneckerMatrices<T = f64>
where
    T: Number,
{
    matrices: Vec<Matrix<T>>,
    rows: usize,
    cols: usize,
}

impl<T> KroneckerMatrices<T>
where
    T: Number,
{
    /// The code below means that `a = a_1 ⊗ a_2`
    /// ```
    /// use opensrdk_linear_algebra::*;
    ///
    /// let a_1 = Matrix::<f64>::new(2, 2);
    /// let a_2 = Matrix::<f64>::new(3, 4);
    /// let a = KroneckerMatrices::new(vec![a_1, a_2]);
    /// ```
    pub fn new(matrices: Vec<Matrix<T>>) -> Self {
        let (rows, cols) = matrices
            .iter()
            .fold((1usize, 1usize), |v, m| (v.0 * m.rows(), v.1 * m.cols()));
        Self {
            matrices,
            rows,
            cols,
        }
    }

    pub fn matrices(&self) -> &[Matrix<T>] {
        &self.matrices
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn eject(self) -> Vec<Matrix<T>> {
        self.matrices
    }

    pub fn prod(&self) -> Matrix<T> {
        // let mut new_matrix =
        //     Matrix::from(self.rows, vec![T::one(); self.rows * self.cols]).unwrap();
        let bigp = self.matrices.len();

        // let mut row_block = 1;
        // let mut col_block = 1;

        // for p in (0..bigp).rev() {
        //     for j in 0..self.cols {
        //         for i in 0..self.rows {
        //             new_matrix[j][i] *= self.matrices[p][j / col_block % self.matrices[p].cols()]
        //                 [i / row_block % self.matrices[p].rows()];
        //         }
        //     }
        //     row_block *= self.matrices[p].rows();
        //     col_block *= self.matrices[p].cols();
        // }

        let prod_elem = |a: Matrix<T>, b: Matrix<T>| {
            let rows_a = a.rows();
            let cols_a = a.cols();
            let rows_b = b.rows();
            let cols_b = b.cols();
            let elems = (0..rows_a * rows_b)
                .into_iter()
                .map(|j| {
                    let rows_mod = j % rows_b;
                    (0..cols_a * cols_b)
                        .into_iter()
                        .map(|i| {
                            let cols_mod = i % cols_b;
                            let elem = a[((i - cols_mod) / cols_a, (j - rows_mod) / rows_a)]
                                * b[(cols_mod, rows_mod)];
                            elem
                        })
                        .collect::<Vec<T>>()
                })
                .collect::<Vec<Vec<T>>>()
                .concat();
            Matrix::from(rows_a * cols_b, elems)
        };

        let mut new_matrix = self.matrices[0].clone();
        for p in (1..bigp) {
            let matrix = prod_elem(new_matrix, self.matrices[p].clone()).unwrap();
            new_matrix = matrix;
        }

        new_matrix
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: Vec<f64>) -> Result<Vec<f64>, MatrixError> {
        let n = v.len();

        if self.cols != n {
            return Err(MatrixError::DimensionMismatch);
        }

        let bigp = self.matrices.len();
        let mut u = v.col_mat();

        for p in (0..bigp).rev() {
            let bigu_rows = self.matrices[p].cols();
            let bigu = u.reshape(bigu_rows);
            let k_bigu = &self.matrices[p] * bigu;

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

    #[test]
    fn it_works2() {
        let a = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let b = mat![
            1.0, 2.0;
            3.0, 4.0
        ];

        let prod_elem = |a: Matrix<f64>, b: Matrix<f64>| {
            let rows_a = a.rows();
            let cols_a = a.cols();
            let rows_b = b.rows();
            let cols_b = b.cols();

            let elems = (0..rows_a * rows_b)
                .into_iter()
                .map(|j| {
                    let rows_mod = j % rows_b;
                    (0..cols_a * cols_b)
                        .into_iter()
                        .map(|i| {
                            let cols_mod = i % cols_b;
                            let elem = a[((i - cols_mod) / cols_a, (j - rows_mod) / rows_a)]
                                * b[(cols_mod, rows_mod)];
                            elem
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>()
                .concat();
            Matrix::from(rows_a * rows_b, elems)
        };

        let d = prod_elem(a, b);

        println!("d {:#?}", d);
    }
}
