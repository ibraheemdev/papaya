# `papaya`

A fast and ergonomic concurrent hash-table for read-heavy workloads.

- An ergonomic lock-free API — no more deadlocks!
- Powerful atomic operations.
- Seamless usage in async contexts.
- Extremely scalable, low-latency reads (see [performance](#performance)).
- Predictable latency across all operations.
- Efficient memory usage, with garbage collection powered by [`seize`].

See [the documentation](https://docs.rs/papaya/latest) for usage details.

## Performance

`papaya` is built with read-heavy workloads in mind. As such, read operations are extremely high throughput and provide consistent performance that scales with concurrency, meaning `papaya` will excel in workloads where reads are more common than writes. In write heavy workloads, `papaya` will still provide competitive performance despite not being it's primary use case. See the [benchmarks] for details.

`papaya` aims to provide predictable and consistent latency across all operations. Most operations are lock-free, and those that aren't only block under rare and constrained conditions. `papaya` also features [incremental resizing](ResizeMode). Predictable latency is an important part of performance that doesn't often show up in benchmarks, but has significant implications for real-world usage.

[benchmarks]: ./BENCHMARKS.md
