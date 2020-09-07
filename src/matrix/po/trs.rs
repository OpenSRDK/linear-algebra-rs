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
    pub fn potrs(&self, bt: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != bt.cols {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut bt = bt;

        unsafe {
            dpotrs(
                'U' as u8,
                n,
                bt.rows as i32,
                &self.elems,
                n,
                &mut bt.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(bt),
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
    pub fn potrs(&self, bt: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != bt.cols {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let mut info = 0;

        let n = n as i32;
        let mut bt = bt;

        unsafe {
            zpotrs(
                'U' as u8,
                n,
                bt.rows as i32,
                &self.elems,
                n,
                &mut bt.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(bt),
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
        let bt = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        let l = a.potrf().unwrap();
        let x_t = l.potrs(bt).unwrap();

        println!("{:#?}", x_t);
        // assert_eq!(x_t[0][0], 0.0);
        // assert_eq!(x_t[0][1], 1.0);
        // assert_eq!(x_t[1][0], 5.0 / 3.0 - 1.0);
        // assert_eq!(x_t[1][1], 5.0 / 3.0);
    }
}
