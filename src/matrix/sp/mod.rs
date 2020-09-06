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
    rows: usize,
    cols: usize,
    elems: HashMap<(usize, usize), T>,
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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn elems(self) -> HashMap<(usize, usize), T> {
        self.elems
    }

    pub fn elems_ref(&self) -> &HashMap<(usize, usize), T> {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut HashMap<(usize, usize), T> {
        &mut self.elems
    }
}
