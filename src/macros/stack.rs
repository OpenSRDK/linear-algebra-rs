use crate::{macros::sub_matrix::SubMatrix, matrix::Matrix, number::Number, types::Standard};

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

pub enum Stack<'a, U: Number> {
    Only(Box<dyn SubMatrix<U> + 'a>),
    Horizontal(Vec<Stack<'a, U>>),
    Vertical(Vec<Stack<'a, U>>),
}

impl<'a, U> Stack<'a, U>
where
    U: Number,
{
    pub fn matrix(&self) -> Matrix<Standard, U> {
        let size = self.size();
        let mut matrix = Matrix::<Standard, U>::zeros(size.0, size.1);
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
                        panic!("different dimensions")
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
                        panic!("different dimensions")
                    }

                    rows += size.0;
                }

                (rows, columns)
            }
        }
    }

    fn transcript(&self, matrix: &mut Matrix<Standard, U>, i: usize, j: usize) -> (usize, usize) {
        match self {
            Stack::Only(sub_matrix) => {
                let size = sub_matrix.size();

                for k in 0..size.0 {
                    for l in 0..size.1 {
                        matrix[i + k][j + l] = sub_matrix.index(i + k, j + l);
                    }
                }

                size
            }
            Stack::Horizontal(vec) => {
                let mut size = (0usize, 0usize);

                for e in vec.iter() {
                    let s = e.transcript(matrix, i + size.0, j);
                    size.0 += s.0;
                    size.1 = s.1;
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
