
#[derive(Default)]
struct OrderedTile<T> {
    items: Vec<T>,
    order: Vec<usize>,
}

impl<T> OrderedTile<T> {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn flush(&mut self) {
        self.items.clear();
        self.order.clear();
    }

    pub fn take(&mut self, i: usize) -> T {
        assert!(i < self.len() && self.len() > 0);
        let idx = self.order.iter().find_position(|x| **x == self.len() - 1).unwrap().0;
        self.order.remove(idx);
        self.order.iter_mut().for_each(|x| if *x >= i { *x += 1; });
        self.items.remove(i)
    }

    pub fn insert(&mut self, i: usize, x: T) {
        assert!(i < self.len());
        self.order.iter_mut().for_each(|x| if *x >= i { *x += 1; });
        self.order.push(self.items.len());
        self.items.insert(i, x);
    }

    pub fn replace(&mut self, i: usize, x: T) {
        assert!(i < self.len());
        self.items[i] = x;
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        let temp = self.order[i];
        self.order[i] = self.order[j];
        self.order[j] = temp;
    }

    pub fn push(&mut self, x: T) {
        self.order.push(self.len());
        self.items.push(x);
    }

    pub fn get(&self, i: usize) -> &T {
        &self.items[self.order[i]]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.items[self.order[i]]
    }
}