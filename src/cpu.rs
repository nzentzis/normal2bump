//! CPU-based implementation of field optimizer
use rayon::prelude::*;

use std::f32::consts::SQRT_2;

use crate::field::{Field, Subfield, VectorField};
use crate::OptimizeParams;

const WORK_ITEM_SIZE: usize = 8;

struct ShardData {
    y: usize,
    ymax: usize,
    data: Subfield<f32>,
    error: f32,
}

pub struct Optimizer<'i> {
    /// Optimization parameters
    params: crate::OptimizeParams,

    /// Current output map
    heightmap: Field<f32>,

    /// Input gradients
    gradients: &'i VectorField<2>,

    /// Number of iterations we've done so far
    iters: usize,

    /// Work-item structures for worker threads
    work_items: Vec<ShardData>,
}

impl<'i> Optimizer<'i> {
    /// Construct an optimizer with a prebuilt starting heightmap
    ///
    /// # Panics
    /// This will panic if `gradients` and `map` are of differing size.
    pub fn new_with_map(
        params: OptimizeParams,
        gradients: &'i VectorField<2>,
        map: Field<f32>
    ) -> Self {
        assert_eq!(gradients.size, map.size);

        let (width, height) = gradients.size;
        let num_shards = (height / WORK_ITEM_SIZE).max(1);
        let work_items = (0..num_shards).into_iter()
                        .map(|idx| {
                            let field_sz = if idx == 0 {WORK_ITEM_SIZE+1} else {WORK_ITEM_SIZE+2};
                            let field_ybase = if idx == 0 {0} else {idx*WORK_ITEM_SIZE - 1};
                            ShardData {
                                y: idx*WORK_ITEM_SIZE,
                                ymax: ((idx+1)*WORK_ITEM_SIZE).min(height),
                                data: Subfield {
                                    field: Field::new((width, field_sz), 0.),
                                    base: (0, field_ybase),
                                },
                                error: 0.,
                            }
                        })
                        .collect::<Vec<_>>();

        Self {
            params, gradients, heightmap: map, work_items,
            iters: 0,
        }
    }

    /// Construct an optimizer with an empty heightmap
    pub fn new(params: crate::OptimizeParams, gradients: &'i VectorField<2>) -> Self {
        let heightmap = Field::new(gradients.size, 0.0);
        Self::new_with_map(params, gradients, heightmap)
    }
}

impl crate::Optimizer for Optimizer<'_> {
    /// Step the optimizer forwards by one iteration
    ///
    /// Returns the accumulated error metric over this iteration.
    fn step(&mut self) -> f32 {
        let step = self.params.step;

        self.work_items.par_iter_mut().for_each(|item| {
            item.data.field.data.fill(0.);
            item.error = 0.;
            for y in item.y..item.ymax {
                for x in 0..self.heightmap.size.0 {
                    let target = self.gradients.get(x, y);
                    let current = self.heightmap.grad_at(x, y);

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
                    if (x+1) < self.heightmap.size.0 {
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
                    if (y+1) < self.heightmap.size.0 {
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
                    if (x+1) < self.heightmap.size.0 && y > 0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x+1, y-1) += delta;
                    }

                    // energy xfer down-left
                    let delta = dot_dl * step;
                    if x > 0 && (y+1) < self.heightmap.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x-1, y+1) += delta;
                    }

                    // energy xfer down-right
                    let delta = dot_dr * step;
                    if (x+1) < self.heightmap.size.0 && (y+1) < self.heightmap.size.0 {
                        *item.data.get_mut(x, y) -= delta;
                        *item.data.get_mut(x+1, y+1) += delta;
                    }
                }
            }
        });

        // update based on sharded delta values
        let mut accum_err = 0.;
        for shard in self.work_items.iter() {
            // shards have border regions of 1 row below/above - make sure to account for those
            let ymin = shard.y.saturating_sub(1);
            let ymax = (shard.ymax + 1).min(self.heightmap.size.1);
            for y in ymin..ymax {
                for x in 0..self.heightmap.size.0 {
                    *self.heightmap.get_mut(x, y) += shard.data.get(x, y);
                }
            }
            accum_err += shard.error;
        }

        // re-normalize
        let (min, max) = self.heightmap.data.par_iter()
                        .map(|x| (*x, *x))
                        .reduce(|| (f32::INFINITY, f32::NEG_INFINITY),
                                |(min_a, max_a), (min_b, max_b)| {
                                    (min_a.min(min_b), max_a.max(max_b))
                                });
        self.heightmap.data.par_iter_mut().for_each(|data| {
            *data = (*data - min) / (max - min);
        });

        self.iters += 1;

        accum_err
    }

    fn iters(&self) -> usize {
        self.iters
    }

    fn finish(&mut self) -> Field<f32> {
        std::mem::replace(&mut self.heightmap, Field::new((0, 0), 0.))
    }
}
