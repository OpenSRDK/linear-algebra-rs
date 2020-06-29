use crate::{
    matrix::{operations::identity::identity, Matrix},
    number::Number,
    types::Diagonal,
};

impl<U> Matrix<Diagonal, U>
where
    U: Number,
{
    /// # Inverse
    /// for Diagonal Matrix
    pub fn inv(&self) -> Self {
        let n = self.rows;
        let mut new_matrix = identity(n);

        for i in 0..n {
            new_matrix[i][i] /= self[i][i];
        }

        new_matrix
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = identity::<f64>(2);
        let a_inv = a.inv();

        assert_eq!(a[0][0], a_inv[0][0])
    }
}
