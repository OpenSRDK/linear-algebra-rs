use crate::number::Number;
use std::collections::HashMap;

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
    elems: HashMap<usize, HashMap<usize, T>>,
}

impl<T> SparseMatrix<T>
where
    T: Number,
{
    pub fn new(rows: usize, cols: usize, elems: HashMap<usize, HashMap<usize, T>>) -> Self {
        Self { rows, cols, elems }
    }

    pub fn t(&self) -> Self {
        Self::new(
            self.cols,
            self.rows,
            self.elems
                .iter()
                .flat_map(|(&i, row)| row.iter().map(move |(&j, &value)| (i, j, value)))
                .fold(
                    HashMap::<usize, HashMap<usize, T>>::new(),
                    |mut m: HashMap<usize, HashMap<usize, T>>, (i, j, value): (usize, usize, T)| {
                        m.entry(j).or_insert(HashMap::new()).insert(i, value);

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

    pub fn elems(self) -> HashMap<usize, HashMap<usize, T>> {
        self.elems
    }

    pub fn elems_ref(&self) -> &HashMap<usize, HashMap<usize, T>> {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut HashMap<usize, HashMap<usize, T>> {
        &mut self.elems
    }
}
