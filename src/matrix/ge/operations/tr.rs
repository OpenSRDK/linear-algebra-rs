use crate::{number::Number, Matrix};
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Trace
    pub fn tr(&self) -> T {
        (0..self.rows).into_par_iter().map(|i| self[i][i]).sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            1.0, 3.0;
            2.0, 4.0
        ];
        let at = a.tr();

        assert_eq!(at, 5.0);
    }
}
