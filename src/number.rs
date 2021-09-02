pub use blas::c64;
use std::{
    fmt::Debug,
    iter::Product,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait Number:
    Clone
    + Copy
    + Debug
    + Default
    + PartialEq
    + Send
    + Sized
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div
    + DivAssign
    + Neg<Output = Self>
    + Sum
    + Product
{
    fn one() -> Self;
}
impl Number for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Number for c64 {
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
}
