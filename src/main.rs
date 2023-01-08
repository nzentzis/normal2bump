use std::io::Write;

mod cpu;
mod gpu;

mod field;
use field::*;

const BASE_STEP: f32 = 0.001;

const ERR_WINDOW: usize = 1_000;
const ERR_THRESHOLD: f32 = 1.0e-10;

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
    out_img.save(fname)
}

/// Timing region
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

pub struct OptimizeParams {
    step: f32,
}

fn main() {
    let fname = std::env::args_os().nth(1)
               .expect("no filename given");
    let image = image::io::Reader::open(fname)
               .expect("unable to open image")
               .decode()
               .expect("unable to decode image");
    let rgb8 = image.into_rgb8();

    // compute gradient field
    let mut gradient = Field::new((rgb8.width() as usize, rgb8.height() as usize), [0.0f32; 2]);
    for (x, y, pix) in rgb8.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        *gradient.get_mut(x, y) = gradient_from_normal_color(*pix);
    }

    // build optimizer
    let params = OptimizeParams {
        step: BASE_STEP,
    };
    let optimizer = match std::env::args_os().nth(2) {
        Some(n) => {
            let image = image::io::Reader::open(n)
                       .expect("unable to open image").decode()
                       .expect("unable to decode image")
                       .into_rgb8();
            assert_eq!(image.width(), rgb8.width());
            assert_eq!(image.height(), rgb8.height());

            // initialize heightmap
            let mut out = Field::new((image.width() as usize, image.height() as usize), 0.0f32);
            for (x, y, pix) in image.enumerate_pixels() {
                let val = (pix.0[0] as f32) / 255.;
                *out.get_mut(x as usize, y as usize) = val;
            }

            gpu::Optimizer::new_with_map(params, &gradient, out)
        },
        None => gpu::Optimizer::new(params, &gradient),
    };
    let mut optimizer = match optimizer {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: failed to create optimizer backend: {}", e);
            return;
        }
    };

    let mut err_history = MetricHistory::<ERR_WINDOW>::new();

    let mut r_update = Region::new();

    let stdout = std::io::stdout();
    let mut io_lock = stdout.lock();
    loop {
        // update step
        r_update.begin();
        let accum_err = optimizer.step();
        r_update.end();
        err_history.push(accum_err);

        let iters = optimizer.iters();
        if iters % 100 == 0 {
            let (min, max) = err_history.bounds();
            let err_delta = max - min;
            write!(io_lock,
                "\r{iters:10} {err_delta:10} | update: {r_update}    "
            ).unwrap();
            io_lock.flush().unwrap();

            if err_history.is_full() && err_delta < ERR_THRESHOLD {
                break;
            }
        }

        if accum_err < 0.5 {
            break;
        }
    }

    save_heightmap("final.png", &optimizer.finish()).unwrap();
}
