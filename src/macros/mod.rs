pub mod stack;
pub mod sub_matrix;
pub mod zeros;

#[macro_export]
macro_rules! mat {
    () => {
        {
            Matrix::new(0, 0, vec![])
        }
    };
    ($($e: expr),+) => {
        {
            use $crate::stack;
            stack!($($e),+).matrix()
        }
    };
    ($($($e: expr),+);+) => {
        {
            use $crate::stack;
            stack!($($($e),+);+).matrix()
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 0.0, 1.0;
            0.0, 1.0, 0.0
        );

        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[0][1], 0.0);
        assert_eq!(a[1][0], 0.0);
        assert_eq!(a[1][2], 0.0);
    }
}
