// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod thread_id;

use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::atomic::{self, AtomicBool, AtomicPtr, AtomicUsize, Ordering};

const BUCKETS: usize = (usize::BITS + 1) as usize;

pub struct ThreadLocal<T: Send> {
    buckets: [AtomicPtr<Entry<T>>; BUCKETS],
    pub threads: AtomicUsize,
}

struct Entry<T> {
    present: AtomicBool,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        unsafe {
            if *self.present.get_mut() {
                ptr::drop_in_place((*self.value.get()).as_mut_ptr());
            }
        }
    }
}

unsafe impl<T: Send> Sync for ThreadLocal<T> {}

impl<T> ThreadLocal<T>
where
    T: Send,
{
    pub fn with_capacity(capacity: usize) -> ThreadLocal<T> {
        let allocated_buckets = capacity
            .checked_sub(1)
            .map(|c| (usize::BITS as usize) - (c.leading_zeros() as usize) + 1)
            .unwrap_or(0);

        let mut buckets = [ptr::null_mut(); BUCKETS];
        let mut bucket_size = 1;
        for (i, bucket) in buckets[..allocated_buckets].iter_mut().enumerate() {
            *bucket = allocate_bucket::<T>(bucket_size);

            if i != 0 {
                bucket_size <<= 1;
            }
        }

        ThreadLocal {
            // safety: `AtomicPtr` has the same representation as a pointer
            buckets: unsafe { mem::transmute(buckets) },
            threads: AtomicUsize::new(0),
        }
    }

    pub fn get_or(&self, create: impl Fn() -> T) -> &T {
        let thread = thread_id::get();

        let bucket = unsafe { self.buckets.get_unchecked(thread.bucket) };
        let mut bucket_ptr = bucket.load(Ordering::Acquire);

        if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket(thread.bucket_size);

            match bucket.compare_exchange(
                ptr::null_mut(),
                new_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => bucket_ptr = new_bucket,
                // if the bucket value changed (from null), that means
                // another thread stored a new bucket before we could,
                // and we can free our bucket and use that one instead
                Err(other) => unsafe {
                    let _ = Box::from_raw(ptr::slice_from_raw_parts_mut(
                        new_bucket,
                        thread.bucket_size,
                    ));

                    bucket_ptr = other;
                },
            }
        }

        unsafe {
            let entry = &*bucket_ptr.add(thread.index);

            // read without atomic operations as only this thread can set the value.
            if (&entry.present as *const _ as *const bool).read() {
                (*entry.value.get()).assume_init_ref()
            } else {
                // insert the new element into the bucket
                entry.value.get().write(MaybeUninit::new(create()));
                entry.present.store(true, Ordering::Release);

                self.threads.fetch_add(1, Ordering::Relaxed);

                // seqcst: synchronize with the fence in `retire`:
                // - if this fence comes first, the thread retiring will see the new thread count
                //   and our entry
                // - if their fence comes first, we will see the new values of any pointers being
                //   retired by that thread
                atomic::fence(Ordering::SeqCst);

                (*entry.value.get()).assume_init_ref()
            }
        }
    }

    #[cfg(test)]
    fn get(&self) -> Option<&T> {
        let thread = thread_id::get();
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);

        if bucket_ptr.is_null() {
            return None;
        }

        unsafe {
            let entry = &*bucket_ptr.add(thread.index);
            // read without atomic operations as only this thread can set the value.
            if (&entry.present as *const _ as *const bool).read() {
                Some((*entry.value.get()).assume_init_ref())
            } else {
                None
            }
        }
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            bucket: 0,
            bucket_size: 1,
            index: 0,
            thread_local: self,
        }
    }
}

impl<T> Drop for ThreadLocal<T>
where
    T: Send,
{
    fn drop(&mut self) {
        let mut bucket_size = 1;

        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let bucket_ptr = *bucket.get_mut();

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                continue;
            }

            unsafe { Box::from_raw(std::slice::from_raw_parts_mut(bucket_ptr, this_bucket_size)) };
        }
    }
}

pub struct Iter<'a, T>
where
    T: Send,
{
    thread_local: &'a ThreadLocal<T>,
    bucket: usize,
    bucket_size: usize,
    index: usize,
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: Send,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // we have to check all the buckets here because we reuse
        // thread IDs. keeping track of the number of values and only
        // yielding that many here wouldn't work, because a new thread
        // could join and be inserted into a middle bucket, and we would
        // yield that instead of an active thread that actually needs to
        // participate in reference counting. yielding extra values is fine,
        // but not yielding all originally active threads is not.
        while self.bucket < BUCKETS {
            let bucket = unsafe {
                self.thread_local
                    .buckets
                    .get_unchecked(self.bucket)
                    .load(Ordering::Acquire)
            };

            if !bucket.is_null() {
                while self.index < self.bucket_size {
                    let entry = unsafe { &*bucket.add(self.index) };
                    self.index += 1;
                    if entry.present.load(Ordering::Acquire) {
                        return Some(unsafe { &*(*entry.value.get()).as_ptr() });
                    }
                }
            }

            if self.bucket != 0 {
                self.bucket_size <<= 1;
            }

            self.bucket += 1;
            self.index = 0;
        }

        None
    }
}

fn allocate_bucket<T>(size: usize) -> *mut Entry<T> {
    Box::into_raw(
        (0..size)
            .map(|_| Entry::<T> {
                present: AtomicBool::new(false),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect(),
    ) as *mut _
}

#[cfg(test)]
#[allow(clippy::redundant_closure)]
mod tests {
    use super::ThreadLocal;
    use std::cell::RefCell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;
    use std::thread;

    fn make_create() -> Arc<dyn Fn() -> usize + Send + Sync> {
        let count = AtomicUsize::new(0);
        Arc::new(move || count.fetch_add(1, Relaxed))
    }

    #[test]
    fn same_thread() {
        let create = make_create();
        let tls = ThreadLocal::with_capacity(1);
        assert_eq!(None, tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
    }

    #[test]
    fn different_thread() {
        let create = make_create();
        let tls = Arc::new(ThreadLocal::with_capacity(1));
        assert_eq!(None, tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());

        let tls2 = tls.clone();
        let create2 = create.clone();
        thread::spawn(move || {
            assert_eq!(None, tls2.get());
            assert_eq!(1, *tls2.get_or(|| create2()));
            assert_eq!(Some(&1), tls2.get());
        })
        .join()
        .unwrap();

        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
    }

    #[test]
    fn iter() {
        let tls = Arc::new(ThreadLocal::with_capacity(1));
        tls.get_or(|| Box::new(1));

        let tls2 = tls.clone();
        thread::spawn(move || {
            tls2.get_or(|| Box::new(2));
            let tls3 = tls2.clone();
            thread::spawn(move || {
                tls3.get_or(|| Box::new(3));
            })
            .join()
            .unwrap();
            drop(tls2);
        })
        .join()
        .unwrap();

        let tls = Arc::try_unwrap(tls).unwrap_or_else(|_| panic!("."));

        let mut v = tls.iter().map(|x| **x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1, 2, 3], v);
    }

    #[test]
    fn iter_snapshot() {
        let tls = Arc::new(ThreadLocal::with_capacity(1));
        tls.get_or(|| Box::new(1));

        let iterator = tls.iter();
        tls.get_or(|| Box::new(2));

        let v = iterator.map(|x| **x).collect::<Vec<i32>>();
        assert_eq!(vec![1], v);
    }

    #[test]
    fn test_drop() {
        let local = ThreadLocal::with_capacity(1);
        struct Dropped(Arc<AtomicUsize>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        local.get_or(|| Dropped(dropped.clone()));
        assert_eq!(dropped.load(Relaxed), 0);
        drop(local);
        assert_eq!(dropped.load(Relaxed), 1);
    }

    #[test]
    fn is_sync() {
        fn foo<T: Sync>() {}
        foo::<ThreadLocal<String>>();
        foo::<ThreadLocal<RefCell<String>>>();
    }
}
