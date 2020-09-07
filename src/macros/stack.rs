use crate::{macros::sub_matrix::SubMatrix, matrix::Matrix, number::Number};

#[macro_export]
macro_rules! stack_v {
    ($e: expr) => {{
        use $crate::macros::stack::Stack;
        Stack::Vertical($e)
    }};
}

#[macro_export]
macro_rules! stack_h {
    ($e: expr) => {{
        use $crate::macros::stack::Stack;
        Stack::Horizontal($e)
    }};
}

#[macro_export]
macro_rules! stack {
    ($e: expr) => {
        {
            use $crate::macros::stack::Stack;
            Stack::Only(Box::new($e))
        }
    };
    ($($e: expr),+) => {
        {
            use $crate::stack_h;
            stack_h!(vec![$(stack!($e)),+])
        }
    };
    ($($($e: expr),+);+) => {
        {
            use $crate::stack_v;
            stack_v!(vec![$(stack!($($e),+)),+])
        }
    };

}

pub enum Stack<'a, T: Number> {
    Only(Box<dyn SubMatrix<T> + 'a>),
    Horizontal(Vec<Stack<'a, T>>),
    Vertical(Vec<Stack<'a, T>>),
}

impl<'a, T> Stack<'a, T>
where
    T: Number,
{
    pub fn matrix(&self) -> Matrix<T> {
        let size = self.size();
        let mut matrix = Matrix::<T>::new(size.0, size.1);
        self.transcript(&mut matrix, 0, 0);

        matrix
    }

    fn size(&self) -> (usize, usize) {
        match self {
            Stack::Only(sub_matrix) => sub_matrix.size(),
            Stack::Horizontal(vec) => {
                let mut rows = 0usize;
                let mut columns = 0usize;

                for e in vec.iter() {
                    let size = e.size();

                    if rows == 0 {
                        rows = size.0;
                    } else if rows != size.0 {
                        panic!("Dimension mismatch.")
                    }

                    columns += size.1;
                }

                (rows, columns)
            }
            Stack::Vertical(vec) => {
                let mut rows = 0usize;
                let mut columns = 0usize;

                for e in vec.iter() {
                    let size = e.size();

                    if columns == 0 {
                        columns = size.1;
                    } else if columns != size.1 {
                        panic!("Dimension mismatch.")
                    }

                    rows += size.0;
                }

                (rows, columns)
            }
        }
    }

    fn transcript(&self, matrix: &mut Matrix<T>, i: usize, j: usize) -> (usize, usize) {
        match self {
            Stack::Only(sub_matrix) => {
                let size = sub_matrix.size();

                for k in 0..size.0 {
                    for l in 0..size.1 {
                        matrix[i + k][j + l] = sub_matrix.index(k, l);
                    }
                }

                size
            }
            Stack::Horizontal(vec) => {
                let mut size = (0usize, 0usize);

                for e in vec.iter() {
                    let s = e.transcript(matrix, i, j + size.1);
                    size.0 = s.0;
                    size.1 += s.1;
                }

                size
            }
            Stack::Vertical(vec) => {
                let mut size = (0usize, 0usize);

                for e in vec.iter() {
                    let s = e.transcript(matrix, i + size.0, j);
                    size.0 += s.0;
                    size.1 = s.1;
                }

                size
            }
        }
    }
}
