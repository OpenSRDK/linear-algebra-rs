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
        let bigp = self.matrices.len();
        let rows = self.matrices[0].rows();
        let cols = self.matrices[0].cols();
        let elems_row = (0..rows.pow(bigp as u32))
            .into_iter()
            .map(|j| {
                (0..cols.pow(bigp as u32))
                    .into_iter()
                    .map(|i| {
                        let elem_a = (0..bigp - 1)
                            .into_iter()
                            .map(|p| {
                                let k = bigp - 1 - p;
                                let row =
                                    ((j - (j % rows.pow(k as u32))) / rows.pow(k as u32)) % rows;
                                let col =
                                    ((i - (i % cols.pow(k as u32))) / cols.pow(k as u32)) % cols;
                                self.matrices[p][(col, row)]
                            })
                            .product::<T>();
                        let elem_b = self.matrices[bigp - 1][(i % cols, j % rows)];
                        elem_a * elem_b
                    })
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .concat();
        let result = Matrix::from(rows.pow(bigp as u32), elems_row).unwrap();
        result
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
            1.0, 2.0, 3.0;
            3.0, 4.0, 5.0;
            0.0, 7.0, 8.0
        ];
        let b = mat![
            1.0, 2.0, 3.0;
            3.0, 4.0, 5.0;
            6.0, 7.0, 5.0
        ];
        let c = mat![
            1.0, 2.0, 3.0;
            3.0, 9.0, 5.0;
            6.0, 7.0, 8.0
        ];
        let abc = KroneckerMatrices::new(vec![a, b, c]);
        let d = abc.prod();
        println!("d {:#?}", d);
    }
}
