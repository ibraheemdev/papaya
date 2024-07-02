#![allow(unstable_name_collisions, clippy::multiple_bound_locations)]
#![doc = include_str!("../README.md")]

mod map;
mod raw;

pub use map::{HashMap, HashMapBuilder, HashMapRef, Iter, Keys, ResizeMode, Values};
pub use seize::Guard;
