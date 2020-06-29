use crate::matrix::Matrix;
use crate::number::Number;
use std::{fmt::Debug, intrinsics::transmute};

pub trait Type: Clone + Copy + Debug + Default + Send + Sync {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Standard;

#[derive(Clone, Copy, Debug, Default)]
pub struct Square;

#[derive(Clone, Copy, Debug, Default)]
pub struct UpperTriangle;

#[derive(Clone, Copy, Debug, Default)]
pub struct LowerTriangle;

#[derive(Clone, Copy, Debug, Default)]
pub struct Diagonal;

#[derive(Clone, Copy, Debug, Default)]
pub struct PositiveDefinite;

#[derive(Clone, Copy, Debug, Default)]
pub struct PositiveSemiDefinite;

impl Type for Standard {}
impl Type for Square {}
impl Type for UpperTriangle {}
impl Type for LowerTriangle {}
impl Type for Diagonal {}
impl Type for PositiveDefinite {}
impl Type for PositiveSemiDefinite {}

impl<T, U> Matrix<T, U>
where
    T: Type,
    U: Number,
{
    pub fn transmute<V: Type>(self) -> Matrix<V, U> {
        unsafe { transmute::<Self, Matrix<V, U>>(self) }
    }

    pub fn transmute_ref<V: Type>(&self) -> &Matrix<V, U> {
        unsafe { transmute::<&Self, &Matrix<V, U>>(self) }
    }
}
