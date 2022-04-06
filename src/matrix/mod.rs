use std::error::Error;

pub mod bd;
pub mod ci;
pub mod di;
pub mod ge;
pub mod gt;
pub mod kr;
pub mod sp_hp;
pub mod ss;
pub mod st;
pub mod to;

#[derive(thiserror::Error, Debug)]
pub enum MatrixError {
    #[error("Dimension mismatch (Matrix)")]
    DimensionMismatch,
    #[error("BLAS routine error. routine: {routine}, info: {info}")]
    BlasRoutineError { routine: String, info: i32 },
    #[error("LAPACK routine error. routine: {routine}, info: {info}")]
    LapackRoutineError { routine: String, info: i32 },
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}
