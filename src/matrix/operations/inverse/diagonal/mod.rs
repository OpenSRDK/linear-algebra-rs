use crate::{
    matrix::{operations::identity::identity, Matrix},
    number::Number,
    types::Diagonal,
};

impl<U> Matrix<Diagonal, U>
where
    U: Number,
{
    pub fn inverse(&self) -> Self {
        let n = self.rows;
        let mut new_matrix = identity(n);

        for i in 0..n {
            new_matrix[i][i] /= self[i][i];
        }

        new_matrix
    }
}
