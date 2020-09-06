use crate::number::Number;
use std::collections::HashMap;

pub mod index;
pub mod mul;
pub mod mul_lhs;
pub mod mul_rhs;

#[derive(Debug)]
pub struct SparseMatrix<T = f64>
where
    T: Number,
{
    pub rows: usize,
    pub cols: usize,
    pub elems: HashMap<(usize, usize), T>,
    default: T,
}

impl<T> SparseMatrix<T>
where
    T: Number,
{
    pub fn new(rows: usize, cols: usize, elems: HashMap<(usize, usize), T>) -> Self {
        Self {
            rows,
            cols,
            elems,
            default: T::default(),
        }
    }

    pub fn t(&self) -> Self {
        Self::new(
            self.cols,
            self.rows,
            self.elems.iter().fold(
                HashMap::<(usize, usize), T>::new(),
                |mut m: HashMap<(usize, usize), T>, (&(i, j), &value)| {
                    m.insert((i, j), value);

                    m
                },
            ),
        )
    }
}