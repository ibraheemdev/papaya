#![allow(dead_code)]
#![deny(unsafe_op_in_unsafe_fn)]
// #![doc = include_str!("../README.md")]

mod cfg;
mod collector;
mod raw;
mod tls;
mod utils;

pub mod reclaim;

pub use collector::{AsLink, Collector, Guard, Link};
