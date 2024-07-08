use std::collections::HashMap;
use std::sync::Mutex;

fn main() {
    println!("=== papaya (incremental) ===");
    p99_insert(papaya::HashMap::new(), |map, i| {
        map.pin().insert(i, ());
    });
    p99_concurrent_insert(papaya::HashMap::new(), |map, i| {
        map.pin().insert(i, ());
    });

    println!("=== papaya (blocking) ===");
    let map = papaya::HashMap::builder()
        .resize_mode(papaya::ResizeMode::Blocking)
        .build();
    p99_insert(map.clone(), |map, i| {
        map.pin().insert(i, ());
    });
    p99_concurrent_insert(map, |map, i| {
        map.pin().insert(i, ());
    });

    println!("=== dashmap ===");
    p99_insert(dashmap::DashMap::new(), |map, i| {
        map.insert(i, ());
    });
    p99_concurrent_insert(dashmap::DashMap::new(), |map, i| {
        map.insert(i, ());
    });

    println!("=== std ===");
    p99_insert(Mutex::new(HashMap::new()), |map, i| {
        map.lock().unwrap().insert(i, ());
    });
    p99_concurrent_insert(Mutex::new(HashMap::new()), |map, i| {
        map.lock().unwrap().insert(i, ());
    });
}

fn p99_insert<T>(map: T, insert: impl Fn(&T, usize)) {
    const ITEMS: usize = 10_000_000;

    let mut max = None;

    for i in 0..ITEMS {
        let now = std::time::Instant::now();
        insert(&map, i);
        let elapsed = now.elapsed();

        if max.map(|max| elapsed > max).unwrap_or(true) {
            max = Some(elapsed);
        }
    }

    println!("p99 insert: {}ms", max.unwrap().as_millis());
}

fn p99_concurrent_insert<T: Sync>(map: T, insert: impl Fn(&T, usize) + Send + Copy) {
    const ITEMS: usize = 2_000_000;

    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..8 {
            let (barrier, map) = (&barrier, &map);
            let handle = s.spawn(move || {
                barrier.wait();

                let mut max = Some(std::time::Instant::now().elapsed());
                for i in 0..ITEMS {
                    let i = (t + 1) * i;

                    let now = std::time::Instant::now();
                    insert(&map, i);
                    let elapsed = now.elapsed();

                    if max.map(|max| elapsed > max).unwrap_or(true) {
                        max = Some(elapsed);
                    }
                }

                max.unwrap()
            });

            handles.push(handle);
        }

        let p99 = handles.into_iter().map(|h| h.join().unwrap()).max();
        println!("p99 concurrent insert: {}ms", p99.unwrap().as_millis());
    });
}
