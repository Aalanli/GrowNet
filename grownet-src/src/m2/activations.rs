use super::GlobalParams;

pub struct WeightedSigmoid {
    pub s: f32,
    pub b: f32,
    pub ds: f32,
    pub db: f32,
    pub x: f32,
    pub y: f32,
}

impl WeightedSigmoid {
    pub fn forward(&mut self, x: f32) -> f32 {
        self.x = x;
        self.y = 1.0 / (1.0 + (self.b * (x - self.s)).exp());
        self.y
    }

    pub fn backward(&mut self, grad: f32) -> f32 {
        let dy_dsigmoid = grad * self.y * (1.0 - self.y);
        let dy_dx = self.b * dy_dsigmoid;
        self.ds += -dy_dx;
        self.db += dy_dsigmoid * (self.x - self.s);
        dy_dx
    }

    pub fn zero_grad(&mut self) {
        self.ds = 0.0;
        self.db = 0.0;
    }

    pub fn apply_grad(&mut self, params: &GlobalParams) {
        self.b -= self.db * params.lr;
        self.s -= self.ds * params.lr;
    }

    pub fn is_underflow(&self, x: f32, eps: f32) -> bool {
        self.s - eps / self.b > x
    }
}
pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}