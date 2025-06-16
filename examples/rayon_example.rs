//! Example of using rayon parallel iterators with papaya HashMap

use papaya::HashMap;
use rayon::prelude::*;

fn main() {
    // Create and populate a HashMap
    let map = HashMap::new();
    let pinned = map.pin_owned();
    
    // Insert some test data
    for i in 0..1000 {
        pinned.insert(i, i * 2);
    }
    
    // Use parallel iterator to sum all values
    let sum: i32 = (&pinned)
        .into_par_iter()
        .map(|(_, v)| *v)
        .sum();
    
    println!("Sum of all values: {}", sum);
    
    // Use parallel iterator to find all even keys
    let even_keys: Vec<_> = (&pinned)
        .into_par_iter()
        .filter(|(k, _)| *k % 2 == 0)
        .map(|(k, _)| *k)
        .collect();
    
    println!("Number of even keys: {}", even_keys.len());
}