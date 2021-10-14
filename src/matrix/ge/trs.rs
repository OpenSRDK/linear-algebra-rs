use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dgetrs, zgetrs};

impl Matrix {
    /// # Solve equation
    ///
    /// with matrix decomposed by getrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn getrs(&self, ipiv: &[i32], b: Matrix) -> Result<Matrix, MatrixError> {
        let n = self.rows();
        if n != self.cols() || n != b.rows {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            dgetrs(
                'N' as u8,
                n,
                b.cols as i32,
                &self.elems,
                n,
                ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dgetrs".to_owned(),
                info,
            }),
        }
    }
}

impl Matrix<c64> {
    /// # Solve equation
    ///
    /// with matrix decomposed by getrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn getrs(&self, ipiv: &[i32], bt: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let n = self.rows();
        if n != self.cols() || n != bt.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;

        let n = n as i32;
        let mut bt = bt;

        unsafe {
            zgetrs(
                'T' as u8,
                n,
                bt.rows as i32,
                &self.elems,
                n,
                ipiv,
                &mut bt.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(bt),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zgetrs".to_owned(),
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
        let a = mat!(
            2.0, 1.0;
            1.0, 1.0
        );
        let b = mat!(
            3.0;
            2.0
        );
        let result = a.clone().getrf().unwrap();
        let x = result.0.getrs(&result.1, b).unwrap();
        let ans = mat!(
            1.0;
            1.0
        );
        assert_eq!(x, ans);
    }
}
