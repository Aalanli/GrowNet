pub mod array_alloc;
pub mod allocator;
pub mod id_algebra;

pub use array_alloc::{
    ArrayAlloc,
    ArrayAllocView,
    ArrayId
};

/// YOLO type level constructs (you only loop once)
pub use id_algebra::{
    Exec,
    is_nan,
    is_infinite,
    is_finite,
    is_normal,
    floor,
    ceil,
    round,
    trunc,
    fract,
    abs,
    signum,
    is_sign_positive,
    is_sign_negative,
    recip,
    powi,
    powf,
    sqrt,
    exp,
    exp2,
    ln,
    log,
    log2,
    log10,
    to_degrees,
    to_radians,
    max,
    min,
    cbrt,
    hypot,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    exp_m1,
    ln_1p,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
};
