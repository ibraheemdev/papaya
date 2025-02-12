use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// A simple lock-free, append-only, stack of pointers.
///
/// This stack is used to defer the reclamation of borrowed entries during
/// an incremental resize, which is relatively rare.
///
/// Alternative, the deletion algorithm could traverse and delete the entry
/// from previous tables to ensure it is unreachable from the root. However,
/// it's not clear whether this is better than an allocation or even lock.
pub struct Stack<T> {
    head: AtomicPtr<Node<T>>,
}

struct Node<T> {
    value: T,
    next: *mut Node<T>,
}

impl<T> Stack<T> {
    /// Create a new `Stack`.
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Add an entry to the stack.
    pub fn push(&self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value,
            next: ptr::null_mut(),
        }));

        loop {
            // Load the head node.
            //
            // `Relaxed` is sufficient here as all reads are through `&mut self`.
            let head = self.head.load(Ordering::Relaxed);

            // Link the node to the stack.
            unsafe { (*node).next = head }

            // Attempt to push the node.
            //
            // `Relaxed` is similarly sufficient here.
            if self
                .head
                .compare_exchange(head, node, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Drain all elements from the stack.
    pub fn drain(&mut self, mut f: impl FnMut(T)) {
        let mut head = *self.head.get_mut();

        while !head.is_null() {
            // Safety: We have `&mut self` and the node is non-null.
            let owned_head = unsafe { Box::from_raw(head) };

            // Drain the element.
            f(owned_head.value);

            // Continue iterating over the stack.
            head = owned_head.next;
        }
    }
}
