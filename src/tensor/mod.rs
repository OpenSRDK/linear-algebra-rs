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
    DimensionMismatch,
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}
