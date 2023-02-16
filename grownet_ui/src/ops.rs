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
    let train_data_writer = std::fs::File::create(path).unwrap();
    bincode::serialize_into(train_data_writer, x).expect("unable to serialize");
}