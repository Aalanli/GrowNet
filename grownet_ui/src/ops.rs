use std::collections::VecDeque;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use bincode;

// removes the first instance where f evaluates true, returns true is anything is removed, false otherwise
pub fn remove_once_if_any<T>(queue: &mut VecDeque<T>, mut f: impl FnMut(&T) -> bool) -> bool {
    let idx = {
        let mut u = -1;
        for (i, r) in queue.iter().enumerate() {
            if f(r) {
                u = i as isize;
                break;
            }
        }
        u
    };
    if idx != -1 {
        queue.remove(idx as usize);
        true
    } else {
        false
    }
}
