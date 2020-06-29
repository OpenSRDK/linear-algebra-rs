pub mod stack;
pub mod sub_matrix;
pub mod zeros;


#[macro_export]
macro_rules! mat {
    [] => {
        {
            Matrix::new(0, 0, vec![])
        }
    };
    [$e: expr] => {
        {
            use $crate::stack;
            stack![$e].matrix()
        }
    };
    [$([$($e: expr),+]);+] => {
        {
            use $crate::stack;
            stack![$([$($e),+]);+].matrix()
        }
    };
    [$($e: expr),+] => {
        {
            use $crate::stack;
            stack![$($e),+].matrix()
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        
    }
}
