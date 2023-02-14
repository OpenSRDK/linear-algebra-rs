use super::SparseMatrix;
use crate::number::Number;
use std::{collections::HashMap, ops::Mul};

fn mul<T>(lhs: &SparseMatrix<T>, rhs: &SparseMatrix<T>) -> SparseMatrix<T>
where
    T: Number,
{
    if lhs.cols != rhs.rows {
        panic!("Dimension mismatch.");
    }

    let elems_orig = lhs
        .elems
        .iter()
        .map(|(&(l_row, l_col), &l)| {
            let elems_orig = rhs
                .elems
                .iter()
                .filter(|(&(r_row, _r_col), &_r)| l_col == r_row)
                .map(|(&(_r_row, r_col), &r)| {
                    let elem = r * l;
                    ((l_row, r_col), elem)
                })
                .collect::<Vec<((usize, usize), T)>>();
            elems_orig
        })
        .collect::<Vec<Vec<((usize, usize), T)>>>()
        .concat();

    let elems_hash = elems_orig.clone().into_iter().collect::<HashMap<_, _>>();

    let elems = elems_hash
        .iter()
        .map(|((row, col), _)| {
            let mut elems_same = elems_orig.clone();
            elems_same.retain(|((row_v, col_v), _value)| (row_v, col_v) == (row, col));
            let value = elems_same
                .iter()
                .map(|((_row, _col), value)| *value)
                .sum::<T>();
            ((*row, *col), value)
        })
        .collect::<HashMap<_, _>>();

    let new_matrix = SparseMatrix::from(lhs.rows, rhs.cols, elems);

    new_matrix
}

impl<T> Mul<SparseMatrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(&self, rhs)
    }
}

impl<T> Mul<SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: SparseMatrix<T>) -> Self::Output {
        mul(self, &rhs)
    }
}

impl<T> Mul<&SparseMatrix<T>> for &SparseMatrix<T>
where
    T: Number,
{
    type Output = SparseMatrix<T>;

    fn mul(self, rhs: &SparseMatrix<T>) -> Self::Output {
        mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let mut a = SparseMatrix::new(3, 2);
        a[(0, 0)] = 1.0;
        a[(2, 1)] = 2.0;
        println!("a {:#?}", a);
        let mut b = SparseMatrix::new(2, 2);
        b[(0, 0)] = 3.0;
        b[(1, 0)] = 4.0;
        println!("b {:#?}", b);
        let c = a * b;
        println!("c {:#?}", c);

        // assert_eq!(c[(0, 0)], 3.0);
        // assert_eq!(c[(2, 0)], 8.0);

        let d = mat![
            1.0, 0.0;
            0.0, 0.0;
            0.0, 2.0
        ];
        let e = mat![
            3.0, 0.0;
            4.0, 0.0];
        println!("row {:#?}", d.rows());
        let f = d.dot(&e);
        println!("f {:#?}", f);
    }
}
