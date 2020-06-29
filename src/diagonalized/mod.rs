use crate::{
    matrix::Matrix,
    number::{c64, Number},
    types::{Diagonal, Square},
};

pub struct Diagonalized<U>(
    pub Matrix<Square, U>,
    pub Matrix<Diagonal, U>,
    pub Matrix<Square, U>,
)
where
    U: Number;

macro_rules! implement {
    {$t: ty} => {
      impl Diagonalized<$t> {
            pub fn inverse(&self) -> Matrix<Square, $t> {
                &self.0 * self.1.inverse() * &self.2
            }

            pub fn determinant(&self) -> $t {
                self.1.determinant()
            }
        }
    };
}

implement! {f64}
implement! {c64}
