use std::marker::PhantomData;
use std::alloc::{self as alloc, Layout};
use std::ptr::{self as ptr, NonNull};
use std::mem;

pub struct Alloc {
    buf: FlatAllocator,
    overflow: Vec<FlatAllocator>,
    overflowed: bool,
    max_overflow: usize,
}

/// Represents the block which a particular piece of memory resides, in Alloc<T>
#[derive(Copy, Clone, Debug)]
pub struct Partition(Option<usize>);

/// Acts as if it is an immutable view over a contiguous region of memory, for safety reasons
/// one can only get the slice via Alloc<T>, this may fail as the context may have been updated
/// as the lifetimes of the context and this id are not tied.
#[derive(Copy, Clone, Debug)]
pub struct AllocId<T> {
    partition: Partition,
    idx: FlatAllocId<T>
}

/// An exclusive mutable view into a context, guaranteed to be unique
pub struct AllocView<'a, T> {
    slice: &'a mut [T],
    part: Partition
}

impl<'a, T> AllocView<'a, T> {
    pub fn slice(&mut self) -> &mut &'a mut [T] {
        &mut self.slice
    }

    pub fn part(&self) -> Partition {
        self.part
    }

    pub fn into_id(self) -> AllocId<T> {
        AllocId::new(self.slice, self.part)
    }

    pub fn destructure(self) -> (&'a mut [T], Partition) {
        (self.slice, self.part)
    }
}

impl<T> AllocId<T> {
    pub fn new(slice: &mut [T], partition: Partition) -> Self {
        Self { partition, idx: FlatAllocId { ptr: NonNull::new(slice.as_ptr() as *mut T).unwrap(), len: slice.len() } }
    }
}

impl Alloc {
    pub fn new() -> Self {
        Alloc { buf: FlatAllocator::new(0), overflow: Vec::new(), overflowed: false, max_overflow: 0 }
    }

    /// Gets the slice from AllocId
    #[inline]
    pub fn borrow_slice<T>(&self, id: AllocId<T>) -> Option<&[T]> {
        if let Some(idx) = id.partition.0 {
            if let Some(block) = self.overflow.get(idx) {
                return block.borrow_slice::<T>(id.idx);
            }
            return None;
        }
        self.buf.borrow_slice(id.idx)
    }

    /// Requests for a mutable slice, returning it along with the partition, which represents the block of
    /// memory which the slice resides in. This is useful for constructing an Id, and retrieving the slice
    #[inline]
    pub fn request_slice<T: Copy>(&self, len: usize) -> AllocView<T> {
        let s = self as *const Self as *mut Self;
        unsafe { // as long as another &mut self is not created, we are fine, right?
            if self.overflowed { // the primary block does not have any more space
                let ptr = Self::request_from_overflow::<T>(s, len); // request from the overflowing blocks
                let slice = std::slice::from_raw_parts_mut(ptr, len);
                let part = Partition(Some(self.overflow.len() - 1));
                return AllocView { slice, part }; // last block is guaranteed to fit
            }
            if let Some(x) = self.buf.request_slice::<T>(len) { 
                return AllocView{ slice: x, part: Partition(None) }; // None represents the primary block
            } else { // the primary block does not have any space
                (*s).overflowed = true;
                // always make sure to be in units of u8
                (*s).max_overflow = self.buf.len.max(len * std::mem::size_of::<T>()) * 2;
                let ptr = Self::request_from_overflow::<T>(s, len);
                return AllocView { slice: std::slice::from_raw_parts_mut(ptr, len), part: Partition(Some(0)) }; // first block is guaranteed to fit
            }
        }
    }

    /// resets the internal buffers and their lengths
    pub fn clear(&mut self) { // guaranteed to have exclusive access to self
        if self.overflow.len() > 0 { // important to only grow the primary buffer if there is actually overflow
            unsafe {
                // each successive overflow block is basically twice the previous
                // and max_overflow is the length of the last overflow block, so summing over all overflow blocks
                // basically twice the length of the last overflow block
                self.buf.buf.grow(self.max_overflow * 2);
            }
        }
        self.overflow.clear(); // remove and free all overflow blocks
        self.overflowed = false;
        self.buf.clear(); // reset the length of the current buffer
    }

    /// here len is the length of [T], while self.len is the length of [u8]
    #[inline]
    unsafe fn request_from_overflow<T: Copy>(s: *mut Self, len: usize) -> *mut T {
        if let Some(x) = (*s).overflow.last() { // if there exists overflow blocks
            if let Some(s) = x.request_slice::<T>(len) {
                return s.as_ptr() as *mut T;
            }
            // otherwise, the last overflowing block has overflowed, make a new overflow block with twice the capacity
            // convert everything to units of u8
            (*s).max_overflow = (len * std::mem::size_of::<T>()).max(x.len) * 2;
        }

        unsafe {
            let new_buf = FlatAllocator::new((*s).max_overflow);
            (*s).overflow.push(new_buf);
            // this is guaranteed to have a slice, as we adjusted max_overflow accordingly
            let slice = (*s).overflow.last().as_ref().unwrap().request_slice::<T>(len).unwrap();
            slice.as_ptr() as *mut T
        }
    }
}

/// Maintains one large chunk of memory, panics if requested size is larger
#[derive(Debug)]
pub struct FlatAllocator {
    buf: FlatBuffer<u8>,
    len: usize,
}

/// Acts as if it is an immutable view over some slice, however, to keep things safe,
/// one can only fetch this slice via an interaction with the FlatAllocator, directly
#[derive(Copy, Clone, Debug)]
pub struct FlatAllocId<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl FlatAllocator {
    pub fn new(size: usize) -> Self {
        let mut buf = FlatBuffer::new();
        unsafe {
            buf.grow(size);
        }
        Self { buf, len: 0 }
    }

    #[inline]
    pub fn request_slice<T: Copy>(&self, len: usize) -> Option<&mut [T]> {
        if self.len + len > self.buf.cap {
            None
        } else {
            unsafe {
                let align = std::mem::align_of::<T>();

                let ptr = self.buf.ptr().add(self.len);
                let pad = ptr.align_offset(align);
                let ptr = ptr.add(pad);
                let slen = &self.len as *const usize as *mut usize;
                *slen += len * std::mem::size_of::<T>() + pad;
                Some(std::slice::from_raw_parts_mut(ptr as *mut T, len))
            }
        }
    }

    #[inline]
    pub fn to_id<T>(&self, slice: &mut [T]) -> Option<FlatAllocId<T>> {
        let len = slice.len();
        let ptr = slice.as_ptr();
        if self.buf.within(ptr as *const u8) {
            Some(FlatAllocId { ptr: NonNull::new(ptr as *mut T).unwrap(), len })
        } else {
            None
        }
    }

    #[inline]
    pub fn borrow_slice<T>(&self, id: FlatAllocId<T>) -> Option<&[T]> {
        let offset = unsafe { 
            let id_ptr = id.ptr.as_ptr();
            let buf_ptr = self.buf.ptr.as_ptr();
            (id_ptr as *mut u8).offset_from(buf_ptr)
        };
        if offset >= 0 && offset as usize + id.len * std::mem::size_of::<T>() <= self.len {
            unsafe { Some(std::slice::from_raw_parts(id.ptr.as_ptr(), id.len)) }
        } else {
            None
        }
    }

    pub fn cap(&self) -> usize {
        self.buf.cap
    }

    #[inline]
    pub fn within<T>(&self, ptr: *const T) -> bool {
        // the only way to get a pointer pointing to self is through the interface
        // which ensures that if a pointer is within the buffer, its whole length is
        // within the buffer
        self.buf.within(ptr as *const u8)
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// A flat buffer into a contiguous chunk of memory that frees itself
#[derive(Debug)]
pub struct FlatBuffer<T> {
    ptr: NonNull<T>,
    cap: usize,
    _marker: PhantomData<T>,
}

impl<T> FlatBuffer<T> {
    pub fn new() -> Self {
        assert!(mem::size_of::<T>() != 0, "No ZSTs");
        Self {
            ptr: NonNull::dangling(),
            cap: 0,
            _marker: PhantomData,
        }
    }

    /// grows the internal storage in addition to the size that it already has
    pub unsafe fn grow(&mut self, size: usize) {
        let new_size = (self.cap + size) as usize;
        self.ptr = grow::<T>(new_size, self.ptr, self.cap);
        self.cap += new_size;
    }

    pub fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn within(&self, ptr: *const T) -> bool {
        let offset = unsafe { ptr.offset_from(self.ptr.as_ptr()) };
        offset >= 0 && offset < self.cap as isize
    }

    pub fn offset(&self, ptr: *const T) -> usize {
        unsafe { ptr.offset_from(self.ptr.as_ptr()) as usize }
    }

    /// current number of elements allocated, useful for future mass allocation/reservation
    pub fn capacity(&self) -> usize {
        self.cap
    }
}

impl<T> Drop for FlatBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            free(&mut self.ptr, self.cap);
        }
    }
}

/// frees the block pointed to by ptr
unsafe fn free<T>(ptr: &mut NonNull<T>, cap: usize) {
    if cap != 0 {
        let ptr = ptr.as_ptr();
        for i in 0..cap {
            ptr::drop_in_place(ptr.add(i));
        }
        let layout = Layout::array::<T>(cap).unwrap();
        unsafe {
            alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

/// grows the specified pointer by size, to cap + size
/// mutably sets the pointer to the new allocated pointer, and cap to the new cap
fn grow<T>(size: usize, ptr: NonNull<T>, cap: usize) -> NonNull<T> {
    let (_new_cap, new_layout) = if cap == 0 {
        (size, Layout::array::<T>(size).unwrap())
    } else {
        let new_cap = cap + size;
        let new_layout = Layout::array::<T>(new_cap).unwrap();
        (new_cap, new_layout)
    };

    assert!(
        new_layout.size() <= isize::MAX as usize,
        "Allocation too large"
    );

    let new_ptr = if cap == 0 {
        unsafe { alloc::alloc(new_layout) }
    } else {
        let old_layout = Layout::array::<T>(cap).unwrap();
        let old_ptr = ptr.as_ptr() as *mut u8;
        unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) }
    };
    // If allocation fails, `new_ptr` will be null, in which case we abort.
    match NonNull::new(new_ptr as *mut T) {
        Some(p) => p,
        None => alloc::handle_alloc_error(new_layout),
    }
}

#[test]
fn test_flat_alloc() {
    let alloc = Alloc::new();
    let mut view = alloc.request_slice::<f32>(15);
    let save_slice = view.slice().to_vec();
    let id = view.into_id();
    let og = alloc.borrow_slice(id).expect("expected slice");
    assert!(og == &save_slice);
}