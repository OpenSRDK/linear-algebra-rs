use super::trf::GTTRF;
use crate::matrix::*;
use lapack::dgttrs;
use lapack::zgttrs;

impl GTTRF {
    /// # Solve equation
    /// with matrix decomposed by gttrf
    /// `Ax = b`
    /// return x
    pub fn gttrs(&self, b: Matrix) -> Result<Matrix, MatrixError> {
        let GTTRF(dl, [d, du, du2], ipiv) = self;
        let n = d.len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe {
            dgttrs(
                'N' as u8,
                n,
                b.cols as i32,
                dl,
                d,
                du,
                du2,
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

impl GTTRF<c64> {
    /// # Solve equation
    /// with matrix decomposed by gttrf
    /// `Ax = b`
    /// return x
    pub fn gttrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let GTTRF::<c64>(dl, [d, du, du2], ipiv) = self;
        let n = d.len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe {
            zgttrs(
                'N' as u8,
                n,
                b.cols as i32,
                dl,
                d,
                du,
                du2,
                ipiv,
                &mut b.elems,
                n,
                &mut info,
            )
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zgttrs".to_owned(),
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
        let result = a.gttrf().unwrap();
        let x = result.gttrs(b).unwrap();
        let ans = mat!(
            2.0;
            1.0;
            1.0
        );
        assert_eq!(x, ans);
    }
}
