use serde::{Serialize, Deserialize};
use strum::{IntoEnumIterator, EnumIter};

use super::ImClassifyDataPoint;


#[derive(Debug, Serialize, Deserialize, EnumIter)]
pub enum ClassificationTransforms {
    Normalize
}
