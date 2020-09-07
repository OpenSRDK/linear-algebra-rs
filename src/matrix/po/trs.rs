use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpotrs, zpotrs};
use std::error::Error;

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            dpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
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
    pub fn potrs(&self, b_t: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            zpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
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
        let b_t = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let l = a.potrf().unwrap();
        let x_t = l.potrs(b_t).unwrap();

        println!("{:#?}", x_t);
        // assert_eq!(x_t[0][0], 0.0);
        // assert_eq!(x_t[0][1], 1.0);
        // assert_eq!(x_t[1][0], 5.0 / 3.0 - 1.0);
        // assert_eq!(x_t[1][1], 5.0 / 3.0);
    }
}
