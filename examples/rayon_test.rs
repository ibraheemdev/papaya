//! Test parallel iteration correctness

use papaya::HashMap;
use rayon::prelude::*;
use std::collections::HashSet;

fn main() {
    // Create and populate a HashMap
    let map = HashMap::new();
    let pinned = map.pin_owned();
    
    println!("Inserting 100 entries...");
    for i in 0..100 {
        pinned.insert(i, i * 2);
    }
    
    // Collect all entries using sequential iterator
    let sequential: HashSet<_> = pinned.iter().map(|(k, v)| (*k, *v)).collect();
    println!("Sequential iterator found {} entries", sequential.len());
    
    // Collect all entries using parallel iterator
    let parallel: HashSet<_> = (&pinned).into_par_iter().map(|(k, v)| (*k, *v)).collect();
    println!("Parallel iterator found {} entries", parallel.len());
    
    // Check they're the same
    if sequential == parallel {
        println!("✓ Sequential and parallel iterators returned the same entries!");
    } else {
        println!("✗ Mismatch between sequential and parallel results!");
        
        let missing_in_parallel: Vec<_> = sequential.difference(&parallel).collect();
        let extra_in_parallel: Vec<_> = parallel.difference(&sequential).collect();
        
        if !missing_in_parallel.is_empty() {
            println!("  Missing in parallel: {:?}", missing_in_parallel);
        }
        if !extra_in_parallel.is_empty() {
            println!("  Extra in parallel: {:?}", extra_in_parallel);
        }
    }
    
    // Test parallel sum
    let seq_sum: i32 = pinned.iter().map(|(_, v)| *v).sum();
    let par_sum: i32 = (&pinned).into_par_iter().map(|(_, v)| *v).sum();
    
    println!("\nSequential sum: {}", seq_sum);
    println!("Parallel sum: {}", par_sum);
    
    if seq_sum == par_sum {
        println!("✓ Sums match!");
    } else {
        println!("✗ Sum mismatch!");
    }
}