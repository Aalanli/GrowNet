
/// The policy in which converts a index of type NIndex to an iterator of indices of
/// type OIndex, this is used for getting the neighbour indices of the current node
/// indexed by NIndex. Typically, for efficiency, the output index type OIndex is an
/// iterator of type usize.
pub trait IndexPolicy<'t> {
    type NIndex;
    type OIndex: Iterator<Item = usize>;
    fn new(dims: &[usize]) -> Self; 
    fn pad() -> usize;
    fn get_index(&'t self, i: Self::NIndex) -> Self::OIndex;
}


/// SequentialPolicy1 only outputs neighbours for node n which are in the future of node n,
/// as in, the graph formed by this policy is acyalic
pub use policies::SequentialPolicy1;


pub mod policies {
    use std::slice::Iter;
    use super::IndexPolicy;
    
    /// Basically a wrapper to simulate std::slice::Map<Iter<'_, usize>, impl Fn(&isize) -> usize
    pub struct PolicyIndexWrapper<'a>(Iter<'a, isize>, isize);
    impl<'a> Iterator for PolicyIndexWrapper<'a> {
        type Item = usize;
        fn next(&mut self) -> Option<Self::Item> {
            Some((self.0.next()? + self.1) as usize)
        }
    }
    
    pub struct SequentialPolicy1 {
        indices: Vec<isize>
    }
    
    
    impl<'t> IndexPolicy<'t> for SequentialPolicy1 {
        type NIndex = usize;
        type OIndex = PolicyIndexWrapper<'t>;
        fn new(dims: &[usize]) -> Self {
            SequentialPolicy1 { indices: forward_policy1(dims) }
        }
        fn pad() -> usize {
            1
        }
        fn get_index(&'t self, offset: Self::NIndex) -> Self::OIndex {
            PolicyIndexWrapper(self.indices.iter(), offset as isize)
        }
    }
    
    
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = Vec::new();
        strides.reserve_exact(dims.len());
        unsafe{ strides.set_len(dims.len()); }
        let mut accum = 1;
        for i in (0..dims.len()).rev() {
            strides[i] = accum;
            accum *= dims[i];
        }
    
        strides    
    }
    
    
    /// Compute the linear index dimensions for which each node visits
    /// given the dimension of the grid
    /// Grids which use this should have a padding of 1, since linear indices
    /// are not inbounds around edges
    fn forward_policy1(dims: &[usize]) -> Vec<isize> {
        let mut indices = vec![];
        let nd = 3usize.pow((dims.len() - 1) as u32);
        indices.reserve_exact(nd);
    
        // recursive helper
        fn helper(n: usize, idx: &mut Vec<isize>, strides: &[usize], cur_offset: isize, cur_dim: usize) {
            if cur_dim == n {
                idx.push(cur_offset);
                return;
            }
    
            helper(n, idx, strides, cur_offset - strides[cur_dim] as isize, cur_dim + 1);
            helper(n, idx, strides, cur_offset, cur_dim + 1);
            helper(n, idx, strides, cur_offset + strides[cur_dim] as isize, cur_dim + 1);
        }
        let strides = compute_strides(dims);
        helper(dims.len(), &mut indices, &strides, strides[0] as isize, 1);
    
        indices
    }
    
    
    #[cfg(test)]
    mod test {
        use super::*;
    
        #[test]
        fn strides_test() {
            let dims = vec![3, 2, 5];
            let strides = compute_strides(&dims);
            
            assert_eq!(strides, vec![10, 5, 1]);
        }
        #[test]
        fn forward_policy1_2d() {
            let dims = vec![3, 4];
            let mut nodes = forward_policy1(&dims);
            nodes.sort();
            assert_eq!(nodes, vec![3, 4, 5])
        }
    
        #[test]
        fn forward_policy1_3d() {
            let dims = vec![3, 4, 5];
            let mut nodes = forward_policy1(&dims);
            nodes.sort();
            let val = vec![
                (1, -1, -1),
                (1, -1,  0),
                (1, -1,  1),
                (1,  0, -1),
                (1,  0,  0),
                (1,  0,  1),
                (1,  1, -1),
                (1,  1,  0), 
                (1,  1,  1)];
            let str = compute_strides(&dims);
            let mut val: Vec<isize> = val.iter()
                .map(|(x, y, z)| x * str[0] as isize + y * str[1] as isize + z * str[2] as isize)
                .collect();
            val.sort();
            assert_eq!(nodes, val)
        }
    }
}