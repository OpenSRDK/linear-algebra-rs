use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpotrs, zpotrs};
use std::error::Error;

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    /// return xt
    pub fn potrs(&self, b: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b.rows {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            dpotrs(
                'L' as u8,
                n,
                b.cols as i32,
                &self.elems,
                n,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpotrs".to_owned(),
                info,
            }
            .into()),
        }
    }
}

impl Matrix<c64> {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    /// return x_t
    pub fn potrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b.rows {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zpotrs(
                'L' as u8,
                n,
                b.cols as i32,
                &self.elems,
                n,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpotrs".to_owned(),
                info,
            }
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            2.0, 1.0;
            1.0, 2.0
        ];
        let b = mat![
            1.0, 3.0;
            2.0, 4.0
        ];
        let l = a.potrf().unwrap();
        let x_t = l.potrs(b).unwrap();

        println!("{:#?}", x_t);
        // assert_eq!(x_t[0][0], 0.0);
        // assert_eq!(x_t[0][1], 1.0);
        // assert_eq!(x_t[1][0], 5.0 / 3.0 - 1.0);
        // assert_eq!(x_t[1][1], 5.0 / 3.0);
    }
}
