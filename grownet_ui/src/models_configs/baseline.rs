pub struct BaselineParams {
    opt: SGDParams,
    epochs: i64,
    
}

pub struct SGDParams {
    momentum: f32,
    dampening: f32,
    wd: f32,
    nesterov: bool
}