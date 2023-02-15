use std::ops::{Index, IndexMut};

use crate::{sparse::SparseTensor, Number, TensorError};

impl<T> Index<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() != self.sizes.len() {
            panic!("{}", TensorError::RankMismatch);
        }
        for (rank, &d) in index.iter().enumerate() {
            if self.sizes[rank] <= d {
                panic!("{}", TensorError::OutOfRange);
            }
        }

        self.elems.get(index).unwrap_or(&self.default)
    }
}

impl<T> IndexMut<&[usize]> for SparseTensor<T>
where
    T: Number,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if index.len() != self.sizes.len() {
            panic!("{}", TensorError::RankMismatch);
        }
        for (rank, &d) in index.iter().enumerate() {
            if self.sizes[rank] <= d {
                panic!("{}", TensorError::OutOfRange);
            }
        }

        self.elems.entry(index.to_vec()).or_default()
    }
}

#[cfg(test)]
mod tests {
    use crate::{sparse::SparseTensor, *};

    #[test]
    fn index() {
        let mut tensor = SparseTensor::new(vec![2, 3]);
        tensor[&[0, 0]] = 1.0;
        tensor[&[1, 1]] = 2.0;
        tensor[&[1, 2]] = 3.0;

        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[0, 1]], 0.0);
        assert_eq!(tensor[&[0, 2]], 0.0);
        assert_eq!(tensor[&[1, 0]], 0.0);
        assert_eq!(tensor[&[1, 1]], 2.0);
        assert_eq!(tensor[&[1, 2]], 3.0);
    }

    #[test]
    fn index_mut() {
        let mut tensor = SparseTensor::new(vec![2, 3]);
        tensor[&[0, 0]] = 1.0;
        tensor[&[1, 1]] = 2.0;
        tensor[&[1, 2]] = 3.0;

        assert_eq!(tensor[&[0, 0]], 1.0);
        assert_eq!(tensor[&[0, 1]], 0.0);
        assert_eq!(tensor[&[0, 2]], 0.0);
        assert_eq!(tensor[&[1, 0]], 0.0);
        assert_eq!(tensor[&[1, 1]], 2.0);
        assert_eq!(tensor[&[1, 2]], 3.0);

        tensor[&[0, 0]] = 0.0;
        tensor[&[1, 1]] = 0.0;
        tensor[&[1, 2]] = 0.0;

        assert_eq!(tensor[&[0, 0]], 0.0);
        assert_eq!(tensor[&[0, 1]], 0.0);
        assert_eq!(tensor[&[0, 2]], 0.0);
        assert_eq!(tensor[&[1, 0]], 0.0);
        assert_eq!(tensor[&[1, 1]], 0.0);
        assert_eq!(tensor[&[1, 2]], 0.0);
    }
}
