use super::SparseMatrix;
use crate::number::Number;
use std::ops::Mul;

fn mul<T>(lhs: &SparseMatrix<T>, rhs: &SparseMatrix<T>) -> SparseMatrix<T>
where
    T: Number,
{
    if lhs.cols != rhs.rows {
        panic!("Dimension mismatch.");
    }
    let mut new_matrix = SparseMatrix::new(lhs.rows, rhs.cols);

    for (&(i, j), &s) in lhs.elems.iter() {
        for (&(_, k), &r) in rhs.elems.iter().filter(|&(&(jr, _), _)| j == jr) {
            let sr = s * r;
            if sr == T::default() {
                continue;
            }

            *new_matrix.elems.entry((i, k)).or_insert(T::default()) += sr;
        }
    }

    println!("lhs_elems {:#?}", lhs.elems.get(&(0, 0)));

    let elems = lhs
        .elems
        .iter()
        .map(|(&(l_row, l_col), &l)| {
            let elems_orig = rhs
                .elems
                .iter()
                .map(|(&(r_row, r_col), &r)| {
                    let mut result = ((l_row, r_col), r);
                    if l_col == r_row {
                        let elem = r * l;
                        result = ((l_row, r_col), elem);
                    } else {
                        result = ((l_row, r_col), r - r);
                    }
                    result
                })
                .collect::<Vec<((usize, usize), T)>>();
            elems_orig
        })
        .collect::<Vec<Vec<((usize, usize), T)>>>()
        .concat();

    println!("elems {:#?}", elems);

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
        let f = d * e;
        println!("f {:#?}", f);
    }
}
