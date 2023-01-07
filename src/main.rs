use std::f32::consts::SQRT_2;
use std::io::Write;

use rayon::prelude::*;

const BASE_STEP: f32 = 0.001;
const WORK_ITEM_SIZE: usize = 8;

const ERR_WINDOW: usize = 3_000;
const ERR_THRESHOLD: f32 = 1.0e-5;

struct Field<T> {
    size: (usize, usize),
    data: Vec<T>,
}

impl<T: Copy> Field<T> {
    fn new(size: (usize, usize), val: T) -> Self {
        Self {
            size,
            data: vec![val; size.0*size.1],
        }
    }

    fn get(&self, x: usize, y: usize) -> T {
        self.data[x + y*self.size.0]
    }

    fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[x + y*self.size.0]
    }
}

impl Field<f32> {
    fn grad_at(&self, x: usize, y: usize) -> [f32; 2] {
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

struct Subfield<T> {
    field: Field<T>,
    base: (usize, usize),
}

impl<T: Copy> Subfield<T> {
    fn get(&self, x: usize, y: usize) -> T {
        self.field.get(x - self.base.0, y - self.base.1)
    }

    fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        assert!(x >= self.base.0, "{} < {}", x, self.base.0);
        assert!(y >= self.base.1, "{} < {}", y, self.base.1);
        let x = x - self.base.0;
        let y = y - self.base.1;
        self.field.get_mut(x, y)
    }
}

/// Given a color in the normal map, return the associated gradient vector
fn gradient_from_normal_color(pixel: image::Rgb<u8>) -> [f32; 2] {
    let x = 2.*(pixel.0[0] as f32 / 255.) - 1.;
    let y = 2.*(pixel.0[0] as f32 / 255.) - 1.;
    let z = pixel.0[0] as f32 / 255.;

    let norm = (x*x + y*y + z*z).sqrt();
    [x / norm, y / norm]
}

/// Save a heightmap
fn save_heightmap(fname: &str, field: &Field<f32>) -> Result<(), image::ImageError> {
    // convert and write
    let mut out_img = image::RgbImage::new(field.size.0 as u32, field.size.1 as u32);
    for (x, y, p) in out_img.enumerate_pixels_mut() {
        let val = field.get(x as usize, y as usize);
        let val = (val * 255.) as u8;
        p.0 = [val; 3];
    }
    Ok(out_img.save(fname)?)
}

struct Region {
    times: Vec<std::time::Duration>,
    next_ptr: usize,
    start: Option<std::time::Instant>,
}

impl Region {
    const WINDOW: usize = 512;

    fn new() -> Self {
        Self {
            times: Vec::with_capacity(Self::WINDOW),
            next_ptr: 0,
            start: None,
        }
    }

    fn begin(&mut self) {
        self.start = Some(std::time::Instant::now());
    }

    fn end(&mut self) {
        let now = std::time::Instant::now();
        if let Some(last) = self.start.take() {
            let dt = now - last;
            if self.times.len() < Self::WINDOW {
                self.times.push(dt);
            } else {
                self.times[self.next_ptr] = dt;
                self.next_ptr = (self.next_ptr + 1) % Self::WINDOW;
            }
        }
    }

    fn average(&self) -> std::time::Duration {
        let sum = self.times.iter().sum::<std::time::Duration>();
        sum / (self.times.len() as u32)
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = self.average();
        if t.as_nanos() < 1000 {
            write!(f, "{} ns", t.as_nanos())
        } else if t.as_micros() < 1000 {
            let ns = t.as_nanos();
            let us = ns / 1000;
            let ns = ns % 1000;
            write!(f, "{}.{:03} us", us, ns)
        } else if t.as_millis() < 1000 {
            let us = t.as_micros();
            let ms = us / 1000;
            let us = us % 1000;
            write!(f, "{}.{:03} ms", ms, us)
        } else {
            let ms = t.as_millis();
            let s = ms / 1000;
            let ms = ms % 1000;
            write!(f, "{}.{:03} s", s, ms)
        }
    }
}

struct MetricHistory<const N: usize> {
    history: Vec<f32>,
    write_ptr: usize,
}

impl<const N: usize> MetricHistory<N> {
    fn new() -> Self {
        Self {
            history: Vec::with_capacity(N),
            write_ptr: 0,
        }
    }

    fn push(&mut self, val: f32) {
        if self.is_full() {
            self.history[self.write_ptr] = val;
            self.write_ptr = (self.write_ptr + 1) % N;
        } else {
            self.history.push(val);
        }
    }

    fn is_full(&self) -> bool {
        self.history.len() == N
    }

    /// Return the smallest and largest values in the history window, as a tuple `(min, max)`
    fn bounds(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for v in self.history.iter() {
            min = min.min(*v);
            max = max.max(*v);
        }

        (min, max)
    }
}

fn main() {
    let fname = std::env::args_os().skip(1).next()
               .expect("no filename given");
    let image = image::io::Reader::open(fname)
               .expect("unable to open image")
               .decode()
               .expect("unable to decode image");
    let rgb8 = image.into_rgb8();

    let base = {
        let fname = std::env::args_os().skip(2).next();
        let image = match fname {
            Some(n) => image::io::Reader::open(n)
                      .expect("unable to open image").decode()
                      .expect("unable to decode image")
                      .into_rgb8(),
            None => image::RgbImage::new(rgb8.width(), rgb8.height()),
        };

        assert_eq!(image.width(), rgb8.width());
        assert_eq!(image.height(), rgb8.height());
        image
    };
    let image = rgb8;

    // compute gradient field
    let mut gradient = Field::new((image.width() as usize, image.height() as usize), [0.0f32; 2]);
    for (x, y, pix) in image.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        *gradient.get_mut(x, y) = gradient_from_normal_color(*pix);
    }

    // construct initial heightmap
    let mut out = Field::new((image.width() as usize, image.height() as usize), 0.0f32);
    for (x, y, pix) in base.enumerate_pixels() {
        let val = (pix.0[0] as f32) / 255.;
        *out.get_mut(x as usize, y as usize) = val;
    }

    // iteratively compute heightmap
    let step = BASE_STEP;
    let mut iters = 0;

    // allocate thread work items
    struct ShardData {
        y: usize,
        ymax: usize,
        data: Subfield<f32>,
        error: f32,
    }
    let num_shards = (image.height() as usize / WORK_ITEM_SIZE).max(1);
    let mut work_items = (0..num_shards).into_iter()
                        .map(|idx| {
                            let field_sz = if idx == 0 {WORK_ITEM_SIZE+1} else {WORK_ITEM_SIZE+2};
                            let field_ybase = if idx == 0 {0} else {idx*WORK_ITEM_SIZE - 1};
                            ShardData {
                                y: idx*WORK_ITEM_SIZE,
                                ymax: ((idx+1)*WORK_ITEM_SIZE).min(image.height() as usize),
                                data: Subfield {
                                    field: Field::new((image.width() as usize, field_sz), 0.),
                                    base: (0, field_ybase),
                                },
                                error: 0.,
                            }
                        })
                        .collect::<Vec<_>>();

    let mut err_history = MetricHistory::<ERR_WINDOW>::new();

    let mut r_update = Region::new();
    let mut r_join = Region::new();
    let mut r_renorm = Region::new();

    let stdout = std::io::stdout();
    let mut io_lock = stdout.lock();
    loop {
        // update step
        r_update.begin();
        work_items.par_iter_mut().for_each(|item| {
            item.data.field.data.fill(0.);
            item.error = 0.;
            for y in item.y..item.ymax {
                for x in 0..out.size.0 {
                    let target = gradient.get(x, y);
                    let current = out.grad_at(x, y);

                    let err = [target[0] - current[0], target[1] - current[1]];
                    item.error += (1.0 - (target[0]*current[0] + target[1]*current[1])).abs();
                    
                    // shove energy around
                    let dot_ul = err[0]/-SQRT_2 + err[1]/-SQRT_2;
                    let dot_u = -err[1];
                    let dot_ur = err[0]/SQRT_2 + err[1]/-SQRT_2;
                    let dot_l = -err[0];
                    let dot_r = err[0];
                    let dot_dl = err[0]/-SQRT_2 + err[1]/SQRT_2;
                    let dot_d = err[1];
                    let dot_dr = err[0]/SQRT_2 + err[1]/SQRT_2;

                    // energy xfer left
                    let delta = dot_l * step;
                    if x > 0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x-1, y) += delta;
                    }

                    // energy xfer right
                    let delta = dot_r * step;
                    if (x+1) < out.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x+1, y) += delta;
                    }

                    // energy xfer up
                    let delta = dot_u * step;
                    if y > 0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x, y-1) += delta;
                    }

                    // energy xfer down
                    let delta = dot_d * step;
                    if (y+1) < out.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x, y+1) += delta;
                    }

                    // energy xfer up-left
                    let delta = dot_ul * step;
                    if x > 0 && y > 0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x-1, y-1) += delta;
                    }

                    // energy xfer up-right
                    let delta = dot_ur * step;
                    if (x+1) < out.size.0 && y > 0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x+1, y-1) += delta;
                    }

                    // energy xfer down-left
                    let delta = dot_dl * step;
                    if x > 0 && (y+1) < out.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x-1, y+1) += delta;
                    }

                    // energy xfer down-right
                    let delta = dot_dr * step;
                    if (x+1) < out.size.0 && (y+1) < out.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x+1, y+1) += delta;
                    }
                }
            }
        });
        r_update.end();

        // update based on sharded delta values
        r_join.begin();
        let mut accum_err = 0.;
        for shard in work_items.iter() {
            // shards have border regions of 1 row below/above - make sure to account for those
            let ymin = shard.y.saturating_sub(1);
            let ymax = (shard.ymax + 1).min(out.size.1);
            for y in ymin..ymax {
                for x in 0..out.size.0 {
                    *out.get_mut(x, y) += shard.data.get(x, y);
                }
            }
            accum_err += shard.error;
        }
        r_join.end();

        // re-normalize
        r_renorm.begin();
        let (min, max) = out.data.par_iter()
                        .map(|x| (*x, *x))
                        .reduce(|| (f32::INFINITY, f32::NEG_INFINITY),
                                |(min_a, max_a), (min_b, max_b)| {
                                    (min_a.min(min_b), max_a.max(max_b))
                                });
        out.data.par_iter_mut().for_each(|data| {
            *data = (*data - min) / (max - min);
        });
        r_renorm.end();

        err_history.push(accum_err);
        if iters % 50 == 0 {
            let (min, max) = err_history.bounds();
            let err_delta = max - min;
            write!(io_lock,
                "\r{iters:10} {err_delta:10} | update: {r_update}  join: {r_join}  renorm: {r_renorm}"
            ).unwrap();
            io_lock.flush().unwrap();

            if err_history.is_full() {
                if err_delta < ERR_THRESHOLD {
                    break;
                }
            }
        }

        if accum_err < 0.5 {
            break;
        }
        iters += 1;
    }

    save_heightmap("final.png", &out).unwrap();
}
