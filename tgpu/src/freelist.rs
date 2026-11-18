use std::collections::VecDeque;

use parking_lot::Mutex;

#[derive(Debug)]
pub struct FreeList<T> {
    inner: Mutex<FreeListInner<T>>,
}

#[derive(Debug)]
struct FreeListInner<T> {
    data: Vec<Option<T>>,
    free: VecDeque<usize>,
}

impl<T> FreeList<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FreeListInner {
                data: Vec::new(),
                free: VecDeque::new(),
            }),
        }
    }

    /// Insert a value, returning a stable index.
    pub fn insert(&self, value: T) -> usize {
        let mut inner = self.inner.lock();

        if let Some(idx) = inner.free.pop_front() {
            inner.data[idx] = Some(value);
            idx
        } else {
            let idx = inner.data.len();
            inner.data.push(Some(value));
            idx
        }
    }

    /// Remove value at `idx`, returning it if it was present.
    pub fn remove(&self, idx: usize) -> Option<T> {
        let mut inner = self.inner.lock();

        if idx >= inner.data.len() {
            return None;
        }

        if let Some(value) = inner.data[idx].take() {
            inner.free.push_back(idx);
            Some(value)
        } else {
            None
        }
    }

    /// Check if an index currently holds a value.
    pub fn contains(&self, idx: usize) -> bool {
        let inner = self.inner.lock();
        inner.data.get(idx).and_then(|slot| slot.as_ref()).is_some()
    }
}
