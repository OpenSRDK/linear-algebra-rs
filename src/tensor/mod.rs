pub mod matrix;
pub mod sparse;

use crate::Number;
use rand::prelude::*;
use std::{collections::HashMap, error::Error, fmt::Debug};

pub type RankIndex = usize;
pub type RankCombinationId = String;

pub fn generate_rank_combination_id() -> RankCombinationId {
    thread_rng().gen::<u32>().to_string()
}

pub fn generate_rank_combinations(
    rank_pairs: &[[RankIndex; 2]],
) -> [HashMap<RankIndex, String>; 2] {
    let mut rank_combinations = [HashMap::new(), HashMap::new()];
    for rank_pair in rank_pairs.iter() {
        let id = generate_rank_combination_id();
        rank_combinations[0].insert(rank_pair[0], id.to_string());
        rank_combinations[1].insert(rank_pair[1], id.to_string());
    }

    rank_combinations
}

pub fn indices_cartesian_product(sizes: &[usize]) -> Vec<Vec<usize>> {
    sizes
        .iter()
        .fold(Vec::<Vec<usize>>::new(), |accum, &next_size| {
            if accum.is_empty() {
                return (0..next_size).map(|i| vec![i]).collect::<Vec<_>>();
            };
            accum
                .into_iter()
                .flat_map(|acc| {
                    (0..next_size)
                        .map(|i| [&acc[..], &[i]].concat())
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .into_iter()
        .collect()
}

pub trait Tensor<T>: Clone + Debug + PartialEq + Send + Sync
where
    T: Number,
{
    fn rank(&self) -> usize;
    fn size(&self, rank: RankIndex) -> usize;
    fn elem(&self, indices: &[usize]) -> T;
    fn elem_mut(&mut self, indices: &[usize]) -> &mut T;
}

#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Dimension mismatch.")]
    RankMismatch,
    #[error("Out of range.")]
    OutOfRange,
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}

#[cfg(test)]
mod tests {
    #[test]
    fn generate_rank_combinations() {
        use super::generate_rank_combinations;
        let a = generate_rank_combinations(&[[0, 0], [1, 1]]);
        println!("a:{:?}", a);
        let b = generate_rank_combinations(&[[0, 0], [1, 1], [2, 2]]);
        println!("b:{:?}", b);
        assert_eq!(a[0].get(&0).unwrap(), a[1].get(&0).unwrap());
        assert_eq!(a[0].get(&1).unwrap(), a[1].get(&1).unwrap());
        assert_eq!(b[0].get(&0).unwrap(), b[1].get(&0).unwrap());
        assert_eq!(b[0].get(&1).unwrap(), b[1].get(&1).unwrap());
        assert_eq!(b[0].get(&2).unwrap(), b[1].get(&2).unwrap());
    }

    #[test]
    fn indices_cartesian_product() {
        use super::indices_cartesian_product;
        let a = indices_cartesian_product(&[2, 2]);
        let b = indices_cartesian_product(&[2, 3, 4]);
        assert_eq!(a, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);

        assert_eq!(
            b,
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 0, 2],
                vec![0, 0, 3],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 2, 0],
                vec![0, 2, 1],
                vec![0, 2, 2],
                vec![0, 2, 3],
                vec![1, 0, 0],
                vec![1, 0, 1],
                vec![1, 0, 2],
                vec![1, 0, 3],
                vec![1, 1, 0],
                vec![1, 1, 1],
                vec![1, 1, 2],
                vec![1, 1, 3],
                vec![1, 2, 0],
                vec![1, 2, 1],
                vec![1, 2, 2],
                vec![1, 2, 3],
            ]
        );
    }
}
