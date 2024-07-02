# Papaya

A fast and ergonomic concurrent hash-table that features:

- An ergonomic lock-free API — no more deadlocks!
- Powerful atomic operations.
- Seamless usage in async contexts.
- Extremely fast and scalable reads (see [benchmarks]).
- Predictable latency across all operations.
- Efficient memory usage, with garbage collection powered by [`seize`].

## Overview

The top-level crate documentation is organized as follows:

- [Usage](#usage) shows how to interact with the concurrent `HashMap`.
- [Atomic Operations](#atomic-operations) shows how to modify a map atomically.
- [Async Support](#async-support) shows how to use the map in an async context.
- [Advanced Lifetimes](#advanced-lifetimes) explains how to use guards when working with nested types.
- [Performance](#performance) provides details of expected performance characteristics.

## Usage

`papaya` aims to provide an ergonomic API without sacrificing performance. The `HashMap` provided by this crate exposes a lock-free API and can hand out direct references to objects in the map without the need for wrapper types that are clunky and prone to deadlocks. 

However, you can't hold on to references forever due to concurrent removals. Because of this, the `HashMap` API is structured around *pinning*:

```rust,ignore
let map = papaya::HashMap::new();

// Pin the map.
let map = map.pin();
```

Once you create a pin you can access the map just like a standard `HashMap`. The pinned table is similar to a lock guard, so any references that are returned will be tied to the lifetime of the guard. Unlike a lock however, pinning is cheap and can never cause a deadlock.

```rust
let map = papaya::HashMap::new();

// Pin the map.
let map = map.pin();

// Use the map as normal.
map.insert('A', 1);
assert_eq!(map.get(&'A'), Some(&1));
assert_eq!(map.len(), 1);
```

As expected of a concurrent `HashMap`, all operations take a shared reference, allowing the map to be freely pinned and accessed from multiple threads:

```rust
let map = papaya::HashMap::new();

// Use the map from multiple threads.
std::thread::scope(|s| {
    // Insert some values.
    s.spawn(|| {
        let map = map.pin();
        for i in 'A'..='Z' {
            map.insert(i, 1);
        }
    });

    // Remove the values.
    s.spawn(|| {
        let map = map.pin();
        for i in 'A'..='Z' {
            map.remove(&i);
        }
    });

    // Read the values.
    s.spawn(|| {
        for (key, value) in map.pin().iter() {
            println!("{key}: {value}");
        }
    });
});
```

It is important to note that as long as you are holding on to a guard, you are preventing the map from performing garbage collection. Pinning and unpinning the table is relatively cheap but not free, similar to the cost of locking and unlocking an uncontended or lightly contended `Mutex`. Thus guard reuse is encouraged, within reason. See the [`seize`] crate for advanced usage and specifics of the garbage collection algorithm.

## Atomic Operations

TODO

## Async Support

By default, a pinned map guard does not implement `Send` as it is tied to the current thread, similar to a lock. This leads to an issue in work-stealing schedulers as guards are not valid across `.await` points.

To overcome this, you can use an *owned* guard.

```rust,ignore
tokio::spawn(async move {
    // Pin the map with an owned guard.
    let map = map.pin_owned();

    // Hold references across await points.
    let value = map.get(37);
    bar().await;
    println!("{}", value);
});
```

Note that owned guards are more expensive to create than regular guards, so they should only be used if necessary. In the above example, you could instead drop the reference and call `get` a second time after the asynchronous call. A more fitting example involves asynchronous iteration:

```rust,ignore
tokio::spawn(async move {
    for (key, value) in map.pin_owned().iter() {
        tokio::fs::write("db.txt", format!("{key}: {value}\n")).await;
    }
});
```

## Advanced Lifetimes

You may run into issues when you try to return a reference to a map contained within an outer type. For example:

```rust,ignore
pub struct Metrics {
    map: papaya::HashMap<String, Vec<u64>>
}

impl Metrics {
    pub fn get(&self, name: &str) -> Option<&[u64]> {
        // error[E0515]: cannot return value referencing temporary value
        Some(self.map.pin().get(name)?.as_slice())
    }
}
```

This is a similar issue to that of locks, as the guard is created within the method and cannot be referenced outside of it. The solution is to accept a guard in the method directly, tying the lifetime to the caller's stack frame:

```rust
use papaya::Guard;

pub struct Metrics {
    map: papaya::HashMap<String, Vec<u64>>
}

impl Metrics {
    pub fn guard(&self) -> impl Guard + '_ {
        self.map.guard()
    }

    pub fn get<'guard>(&self, name: &str, guard: &'guard impl Guard) -> Option<&'guard [u64]> {
        Some(self.map.get(name, guard)?.as_slice())
    }
}
```

The `Guard` trait supports both local and owned guards. Note the `'guard` lifetime that ties the guard to the returned reference. No complicated wrapper types or closure mapping is necessary.

## Performance

`papaya` is built with read-heavy workloads in mind. As such, reads are extremely scalable and provide consistent performance that scales with concurrency, meaning `papaya` will excel in any workload in which reads are more common than writes. In write heavy workloads, `papaya` will still provide competitive performance despite not being it's primary use case. See the [benchmarks] for details.

`papaya` also aims to provide predictable, consistent latency across all operations. Most operations are lock-free, and those that aren't only block under rare and constrained conditions. `papaya` also features [incremental resizing], meaning operations aren't required to block when resizing the hash-table. Predictable latency is an important part of performance that doesn't often show up in benchmarks, but has significant implications for real-world usage.

[benchmarks]: TOOD
[`seize`]: https://docs.rs/seize/latest
[incremental resizing]: https://docs.rs/papaya/latest/papaya/enum.ResizeMode.html
