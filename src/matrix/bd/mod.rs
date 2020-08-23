use crate::{matrix::Matrix, number::Number};

pub struct BidiagonalMatrix<T = f64>
where
    T: Number,
{
    d: Vec<T>,
    e: Vec<T>,
}

impl<T> BidiagonalMatrix<T>
where
    T: Number,
{
    pub fn new(d: Vec<T>, e: Vec<T>) -> Self {
        Self { d, e }
    }

    pub fn get_elements(self) -> (Vec<T>, Vec<T>) {
        (self.d, self.e)
    }

    pub fn get_matrix(&self, upper: bool) -> Matrix<T> {
        let n = self.d.len();
        let mut mat = Matrix::diag(&self.d);

        if upper {
            for i in 0..n - 1 {
                mat[i][i + 1] = self.e[i];
            }
        } else {
            for i in 0..n - 1 {
                mat[i + 1][i] = self.e[i];
            }
        }

        mat
    }
}
