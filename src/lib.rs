#![allow(unstable_name_collisions)]
#![doc = include_str!("../README.md")]

mod map;
mod raw;

pub use map::{HashMap, HashMapBuilder, HashMapRef, Iter, Keys, ResizeMode, Values};
pub use seize::Guard;
