use std::collections::VecDeque;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use bincode;


// load config files if any
pub fn try_deserialize<T: DeserializeOwned>(x: &mut T, path: &std::path::PathBuf) {
    if path.exists() {
        eprintln!("deserializing {}", path.to_str().unwrap());
        let reader = std::fs::File::open(path).expect("unable to open file");
        match bincode::deserialize_from(reader) {
            Ok(y) => { *x = y; },
            Err(e) => { eprintln!("unable to deserialize\n{}", e); }
        }
    } else {
        eprintln!("{} does not exist", path.to_str().unwrap());
    }
}

pub fn serialize<T: Serialize>(x: &T, path: &std::path::PathBuf) {
    eprintln!("serializing {}", path.to_str().unwrap());
    if !path.parent().expect(&format!("path {} does not have a parent", path.display())).exists() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("failed to create directory for serialize");
    }
    let train_data_writer = std::fs::File::create(path).unwrap();
    bincode::serialize_into(train_data_writer, x).expect("unable to serialize");
}

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
