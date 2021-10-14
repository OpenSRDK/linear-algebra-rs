use crate::{matrix::*, BidiagonalMatrix};
use lapack::dgttrs;

impl BidiagonalMatrix<f64> {
    /// # Solve equation
    /// with matrix decomposed by gttrf
    /// `Ax = b`
    /// return x
    pub fn gttrs(&self, u: &[Vec<f64>; 3], ipiv: &[i32], b: Matrix) -> Result<Matrix, MatrixError> {
        let e = self.e();
        let n = self.d().len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe {
            dgttrs(
                'N' as u8,
                n,
                b.cols as i32,
                &e,
                &u[0],
                &u[1],
                &u[2],
                ipiv,
                &mut b.elems,
                n,
                &mut info,
            )
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dgttrs".to_owned(),
                info,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = TridiagonalMatrix::new(vec![1.0; 2], vec![1.0; 3], vec![1.0; 2]).unwrap();
        let b = mat!(
            3.0;
            4.0;
            2.0
        );
        let result = a.clone().gttrf().unwrap();
        let x = result.0.gttrs(&result.1, &result.2, b).unwrap();
        let ans = mat!(
            2.0;
            1.0;
            1.0
        );
        assert_eq!(x, ans);
    }
}
