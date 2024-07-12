#![no_main]

use libfuzzer_sys::fuzz_target;
use std::collections::hash_map::Entry;

use arbitrary::unstructured::Int;
use arbitrary::{Arbitrary, Unstructured};
use papaya::{Guard, HashMap as PapayaHashMap, HashMapRef};
use std::collections::HashMap as StdHashMap;
use std::hash::{BuildHasher, Hash};
use std::ops::Add;

#[derive(Debug, Arbitrary)]
enum Operation<K, V> {
    Insert(K, V),
    Remove(K),
    Get(K),
    Contains(K),
    Clear,
    Len,
    IsEmpty,
    Update(K, V),
    UpdateOrInsert(K, V, V),
    GetOrInsert(K, V),
    Compute(K),
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    operations: Vec<Operation<u32, u32>>,
}

fn fuzz_hashmap(input: FuzzInput) {
    let mut std_map = StdHashMap::new();
    let papaya_raw = PapayaHashMap::new();
    let papaya_map = papaya_raw.pin();

    for op in input.operations {
        match op {
            Operation::Insert(k, v) => {
                let std_result = std_map.insert(k.clone(), v.clone());
                let papaya_result = papaya_map.insert(k, v);
                assert_eq!(std_result.as_ref(), papaya_result);
            }
            Operation::Remove(k) => {
                let std_result = std_map.remove(&k);
                let papaya_result = papaya_map.remove(&k);
                assert_eq!(std_result.as_ref(), papaya_result);
            }
            Operation::Get(k) => {
                let std_result = std_map.get(&k);
                let papaya_result = papaya_map.get(&k);
                assert_eq!(std_result, papaya_result);
            }
            Operation::Contains(k) => {
                let std_result = std_map.contains_key(&k);
                let papaya_result = papaya_map.contains_key(&k);
                assert_eq!(std_result, papaya_result);
            }
            Operation::Clear => {
                std_map.clear();
                papaya_map.clear();
            }
            Operation::Len => {
                assert_eq!(std_map.len(), papaya_map.len());
            }
            Operation::IsEmpty => {
                assert_eq!(std_map.is_empty(), papaya_map.is_empty());
            }
            Operation::Update(k, v) => {
                let std_result = std_map.get_mut(&k).map(|e| {
                    *e = e.wrapping_add(v);
                    e as &u32
                });
                let papaya_result = papaya_map.update(k, |e| e.wrapping_add(v));
                assert_eq!(std_result, papaya_result);
            }
            Operation::UpdateOrInsert(k, v, default) => {
                let std_result = std_map
                    .entry(k.clone())
                    .and_modify(|e| *e = e.wrapping_add(v))
                    .or_insert(default.clone());
                let papaya_result = papaya_map.update_or_insert(k, |e| e.wrapping_add(v), default);
                assert_eq!(std_result, papaya_result);
            }
            Operation::GetOrInsert(k, v) => {
                let std_result = std_map.entry(k.clone()).or_insert(v.clone());
                let papaya_result = papaya_map.get_or_insert(k, v);
                assert_eq!(std_result, papaya_result);
            }
            Operation::Compute(k) => compute(&mut std_map, &papaya_map, k),
        }
    }

    // Final consistency checks
    for (k, v) in std_map.iter() {
        let papaya_result = papaya_map.get(k);
        assert_eq!(Some(v), papaya_result);
    }
    assert_eq!(std_map.len(), papaya_map.len());
    assert_eq!(std_map.is_empty(), papaya_map.is_empty());
}

fn compute<S, G>(std: &mut StdHashMap<u32, u32>, papaya: &HashMapRef<u32, u32, S, G>, k: u32)
where
    S: BuildHasher,
    G: Guard,
{
    match std.entry(k) {
        Entry::Occupied(mut entry) => {
            let value = entry.get();
            if value % 2 == 0 {
                entry.remove();
            } else {
                *entry.get_mut() = value.wrapping_add(1);
            }
        }
        Entry::Vacant(_) => {
            // Do nothing for non-existent keys
        }
    }

    let compute = |entry: Option<(&u32, &u32)>| match entry {
        // Remove the value if it is even.
        Some((_key, value)) if value % 2 == 0 => papaya::Operation::Remove,

        // Increment the value if it is odd.
        Some((_key, value)) => papaya::Operation::Insert(value.wrapping_add(1)),

        // Do nothing if the key does not exist
        None => papaya::Operation::Abort(()),
    };
    papaya.compute(k, compute);
}

fuzz_target!(|data: FuzzInput| {
    fuzz_hashmap(data);
});
