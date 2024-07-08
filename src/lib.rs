#![allow(unstable_name_collisions, clippy::multiple_bound_locations)]
#![doc = include_str!("../README.md")]

mod map;
mod raw;

pub use map::{
    Compute, HashMap, HashMapBuilder, HashMapRef, Iter, Keys, OccupiedError, Operation, ResizeMode,
    Values,
};
pub use seize::{Collector, Guard};
