use std::f32::consts::SQRT_2;

#[derive(Clone)]
pub struct Field<T> {
    pub size: (usize, usize),
    pub data: Vec<T>,
}

impl<T: Copy> Field<T> {
    pub fn new(size: (usize, usize), val: T) -> Self {
        Self {
            size,
            data: vec![val; size.0*size.1],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[x + y*self.size.0]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[x + y*self.size.0]
    }
}

impl Field<f32> {
    pub fn grad_at(&self, x: usize, y: usize) -> [f32; 2] {
        let cur = self.get(x, y);

        let mut dx = 0.;
        let mut dy = 0.;
        let mut n_dx = 0;
        let mut n_dy = 0;

        let up_y    = if y > 0 { y-1 } else { self.size.1-1 };
        let down_y  = if (y+1) < self.size.1 { y+1 } else { 0 };

        let left_x  = if x > 0 { x-1 } else { self.size.0-1 };
        let right_x = if (x+1) < self.size.0 { x+1 } else { 0 };

        { // up
            let up = self.get(x, up_y);
            dy += cur - up;
            n_dy += 1;
        }
        { // down
            let down = self.get(x, down_y);
            dy += down - cur;
            n_dy += 1;
        }

        { // left
            let left = self.get(left_x, y);
            dx += cur - left;
            n_dx += 1;
        }
        { // right
            let right = self.get(right_x, y);
            dx += right - cur;
            n_dx += 1;
        }

        // up-left
        {
            let ul = self.get(left_x, up_y);
            dx += (cur - ul) / SQRT_2;
            dy += (cur - ul) / SQRT_2;
            n_dx += 1;
            n_dy += 1;
        }

        // up-right
        {
            let ur = self.get(right_x, up_y);
            dx += (ur - cur) / SQRT_2;
            dy += (cur - ur) / SQRT_2;
            n_dx += 1;
            n_dy += 1;
        }

        // down-left
        {
            let dl = self.get(left_x, down_y);
            dx += (cur - dl) / SQRT_2;
            dy += (dl - cur) / SQRT_2;
            n_dx += 1;
            n_dy += 1;
        }

        // down-right
        {
            let dr = self.get(right_x, down_y);
            dx += (dr - cur) / SQRT_2;
            dy += (dr - cur) / SQRT_2;
            n_dx += 1;
            n_dy += 1;
        }

        // compute deltas and return
        [dx / (n_dx as f32), dy / (n_dy as f32)]
    }
}

pub struct Subfield<T> {
    pub field: Field<T>,
    pub base: (usize, usize),
}

impl<T: Copy> Subfield<T> {
    pub fn get(&self, x: usize, y: usize) -> T {
        self.field.get(x - self.base.0, y - self.base.1)
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        assert!(x >= self.base.0, "{} < {}", x, self.base.0);
        assert!(y >= self.base.1, "{} < {}", y, self.base.1);
        let x = x - self.base.0;
        let y = y - self.base.1;
        self.field.get_mut(x, y)
    }
}

pub type VectorField<const N: usize> = Field<[f32; N]>;
