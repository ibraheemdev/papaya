fn p99_insert() {
    let map = papaya::HashMap::new();
    let mut max = None;

    for i in 0..5_000_000 {
        let now = std::time::Instant::now();
        map.pin().insert(i, ());
        let elapsed = now.elapsed();

        if max.map(|max| elapsed > max).unwrap_or(true) {
            max = Some(elapsed);
        }
    }

    println!("insert p99: {}ms", max.unwrap().as_millis());
}

fn p99_concurrent_insert() {
    let map = papaya::HashMap::new();
    let barrier = std::sync::Barrier::new(8);

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..8 {
            let (barrier, map) = (&barrier, &map);
            let handle = s.spawn(move || {
                barrier.wait();

                let mut max = Some(std::time::Instant::now().elapsed());
                for i in 0..1_000_000 {
                    let i = (t + 1) * i;

                    let now = std::time::Instant::now();
                    map.pin().insert(i, ());
                    let elapsed = now.elapsed();

                    if max.map(|max| elapsed > max).unwrap_or(true) {
                        max = Some(elapsed);
                    }
                }

                format!("{}ms", max.unwrap().as_millis())
            });

            handles.push(handle);
        }

        let p99 = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
            .join(", ");

        println!("concurrent insert p99: {}", p99);
    });
}

fn main() {
    p99_insert();
    p99_concurrent_insert();
}
