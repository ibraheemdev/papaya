# `papaya`

[<img alt="crates.io" src="https://img.shields.io/crates/v/papaya?style=for-the-badge" height="25">](https://crates.io/crates/papaya)
[<img alt="github" src="https://img.shields.io/badge/github-papaya-blue?style=for-the-badge" height="25">](https://github.com/ibraheemdev/papaya)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/papaya?style=for-the-badge" height="25">](https://docs.rs/papaya)

A fast and ergonomic concurrent hash-table for read-heavy workloads.

See [the documentation](https://docs.rs/papaya/latest) to get started.

## Features

- An ergonomic lock-free API — no more deadlocks!
- Powerful atomic operations.
- Seamless usage in async contexts.
- Extremely scalable, low-latency reads (see [performance](#performance)).
- Predictable latency across all operations.
- Efficient memory usage, with garbage collection powered by [`seize`].

## Performance

`papaya` is built with read-heavy workloads in mind. As such, read operations are extremely high throughput and provide consistent performance that scales with concurrency, meaning `papaya` will excel in workloads where reads are more common than writes. In write heavy workloads, `papaya` will still provide competitive performance despite not being it's primary use case. See the [benchmarks] for details.

`papaya` aims to provide predictable and consistent latency across all operations. Most operations are lock-free, and those that aren't only block under rare and constrained conditions. `papaya` also features [incremental resizing]. Predictable latency is an important part of performance that doesn't often show up in benchmarks, but has significant implications for real-world usage.

[benchmarks]: ./BENCHMARKS.md
[`seize`]: https://github.com/ibraheemdev/seize
[incremental resizing]: https://docs.rs/papaya/latest/papaya/enum.ResizeMode.html
