use clap::Parser;
use anyhow::Context;

use std::io::Write;
use std::path::PathBuf;

mod cpu;
mod gpu;

mod field;
use field::*;

const ERR_WINDOW: usize = 1_000;
const ERR_THRESHOLD: f32 = 1.0e-10;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ComputeBackend { Cpu, Gpu, Auto }

impl ComputeBackend {
    /// Use the given parameters to construct an Optimizer matching this specification
    fn build<'g>(
        &self,
        params: OptimizeParams,
        grad: &'g VectorField<2>,
        seed: Option<Field<f32>>
    ) -> anyhow::Result<Box<dyn Optimizer + 'g>> {
        match self {
            Self::Cpu => {
                let opt = match seed {
                    Some(s) => cpu::Optimizer::new_with_map(params, grad, s),
                    None    => cpu::Optimizer::new(params, grad),
                };
                Ok(Box::new(opt))
            },
            Self::Gpu => {
                let opt = match seed {
                    Some(s) => gpu::Optimizer::new_with_map(params, grad, s)?,
                    None    => gpu::Optimizer::new(params, grad)?,
                };
                Ok(Box::new(opt))
            },
            Self::Auto => {
                let par = params.clone();
                let opt = match seed.as_ref() {
                    Some(s) => gpu::Optimizer::new_with_map(par, grad, s.clone()),
                    None    => gpu::Optimizer::new(par, grad),
                };
                Ok(opt.map(|x: gpu::Optimizer| -> Box<dyn Optimizer + 'g> { Box::new(x) })
                      .unwrap_or_else(move |_| Box::new(match seed {
                          Some(s) => cpu::Optimizer::new_with_map(params, grad, s),
                          None    => cpu::Optimizer::new(params, grad),
                      })))
            },
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_enum, default_value_t=ComputeBackend::Auto)]
    backend: ComputeBackend,

    /// Size of each error-diffusion step
    #[arg(short, long, default_value_t=0.001)]
    step: f32,

    /// Error derivative at which to stop iterating
    #[arg(short, long, default_value_t=ERR_THRESHOLD)]
    threshold: f32,

    /// Disable progress display
    #[arg(short, long)]
    quiet: bool,

    /// Normal map to convert
    input: PathBuf,

    /// Path to output file, or `{input_name}_bump.png` if unspecified
    output: Option<PathBuf>,

    /// Seed image to start from, instead of starting from an empty heightmap
    seed: Option<PathBuf>,
}

trait Optimizer {
    /// Step the optimizer forwards by one iteration
    ///
    /// Returns the accumulated error metric over this iteration.
    fn step(&mut self) -> f32;

    /// Get the number of completed steps
    fn iters(&self) -> usize;

    /// Return the final heightmap
    ///
    /// This may invalidate the optimizer. Stepping after calling this function should panic.
    fn finish(&mut self) -> Field<f32>;
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
fn save_heightmap(fname: &std::path::Path, field: &Field<f32>) -> Result<(), image::ImageError> {
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

#[derive(Clone)]
pub struct OptimizeParams {
    step: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // TODO: don't truncate higher-precision normal maps
    let normal = image::io::Reader::open(&args.input)
                .context("Failed to open input file")?
                .decode().context("Unable to decode input normal map")?
                .into_rgb8();

    // compute gradient field
    let mut gradient = Field::new((normal.width() as usize, normal.height() as usize), [0.0f32; 2]);
    for (x, y, pix) in normal.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        *gradient.get_mut(x, y) = gradient_from_normal_color(*pix);
    }

    // build optimizer
    let seed = args.seed.map(|s| image::io::Reader::open(s)
                                .context("Failed to open seed image")
                                .and_then(|img| img.decode()
                                                   .context("Failed to decode seed image"))
                                .map(|img| img.into_rgb8()))
              .transpose()?
              .map(|seed| {
                  let mut out = Field::new((seed.width() as usize, seed.height() as usize), 0.);
                  for (x, y, pix) in seed.enumerate_pixels() {
                      let val = (pix.0[0] as f32) / 255.;
                      *out.get_mut(x as usize, y as usize) = val;
                  }
                  out
              });
    let mut optimizer = args.backend.build(
        OptimizeParams {
            step: args.step,
        },
        &gradient,
        seed
    ).context("Failed to create optimizer backend")?;

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

            if !args.quiet {
                write!(io_lock,
                    "\r{iters:10} {err_delta:10} | update: {r_update}    "
                ).unwrap();
                io_lock.flush().unwrap();
            }

            if err_history.is_full() && err_delta < args.threshold {
                break;
            }
        }

        if accum_err < 0.5 {
            break;
        }
    }

    let out_path = args.output
                  .unwrap_or_else(|| {
                      let mut res = args.input.clone();
                      let mut name = args.input.file_stem()
                                    .unwrap_or_else(|| "input".as_ref())
                                    .to_owned();
                      name.push("_bump.png");

                      res.set_file_name(name);
                      res
                  });
    save_heightmap(&out_path, &optimizer.finish())
        .context("Failed to save heightmap image")?;
    Ok(())
}
