//! GPU compute implementation of field optimizer
use anyhow::{Result, Context};
use pollster::FutureExt;

use crate::field::{Field, Subfield, VectorField};
use crate::OptimizeParams;

const WG_SIZE: usize = 8;

struct ComputeContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl ComputeContext {
    fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        let adapter = instance.enumerate_adapters(wgpu::Backends::PRIMARY).next()
                     .ok_or_else(|| anyhow::anyhow!("Failed to retrieve compute adapter"))?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
            label: None,
        }, None).block_on().context("Failed to retrieve compute device")?;

        Ok(Self {
            device, queue,
        })
    }
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct ParamsKernel {
    width: u32,
    height: u32,
    step: f32,
    sqrt2: f32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Summary {
    min: f32,
    max: f32,
    total_error: f32,
}

pub struct Optimizer<'i> {
    /// Optimization parameters
    params: crate::OptimizeParams,

    /// Current output map
    heightmap: Field<f32>,

    /// Input gradients
    gradients: &'i VectorField<2>,

    /// Compute context
    context: ComputeContext,

    /// Uniform data buffer
    buf_params: wgpu::Buffer,

    /// Input gradients
    buf_gradients: wgpu::Buffer,

    /// Current output map buffer
    buf_heightmap: wgpu::Buffer,

    /// Error gradient buffer
    buf_errors: wgpu::Buffer,

    /// Summary data buffer
    buf_summary: wgpu::Buffer,

    /// Mappable summary buffer
    buf_summary_map: wgpu::Buffer,

    bind_group: wgpu::BindGroup,

    /// Compute pipeline for error computation
    pl_error: wgpu::ComputePipeline,

    /// Compute pipeline for energy diffusion
    pl_diffuse: wgpu::ComputePipeline,

    /// Compute pipeline for normalizing outputs
    pl_normalize: wgpu::ComputePipeline,

    /// Compute pipeline for summarizing metrics (error metric, min/max)
    pl_summarize: wgpu::ComputePipeline,
    pl_summarize_y: wgpu::ComputePipeline,

    /// Workgroup grid dimensions
    grid_dim: (u32, u32),

    /// Number of iterations we've done so far
    iters: usize,
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
    ) -> Result<Self> {
        assert_eq!(gradients.size, map.size);

        let context = ComputeContext::new()?;
        let plane_cells = gradients.size.0 * gradients.size.1;
        let plane_size = (plane_cells * std::mem::size_of::<f32>()) as u64;

        // allocate working buffers
        let buf_params = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<ParamsKernel>() as u64,
            mapped_at_creation: false,
        });
        let buf_heightmap = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heightmap"),
            usage: wgpu::BufferUsages::STORAGE |
                   wgpu::BufferUsages::COPY_DST |
                   wgpu::BufferUsages::COPY_SRC,
            size: plane_size,
            mapped_at_creation: false,
        });
        let buf_gradients = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gradients"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: plane_size * 2,
            mapped_at_creation: false,
        });
        let buf_errors = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("errors"),
            usage: wgpu::BufferUsages::STORAGE,
            size: plane_size * 2,
            mapped_at_creation: false,
        });
        let buf_summary = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("summary"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: std::mem::size_of::<Summary>() as u64,
            mapped_at_creation: false,
        });
        let buf_summary_map = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mappable summary"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            size: std::mem::size_of::<Summary>() as u64,
            mapped_at_creation: false,
        });

        // load compute module
        let module = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_kernels.wgsl").into()),
        });

        // build compute pipelines
        let pl_group = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    count: None,
                },
            ]
        });

        let pl_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&pl_group],
            push_constant_ranges: &[],
        });
        let pl_error = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("error computation"),
            module: &module,
            entry_point: "compute_errors",
            layout: Some(&pl_layout),
        });
        let pl_diffuse = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("energy diffusion"),
            module: &module,
            entry_point: "energy_diffusion",
            layout: Some(&pl_layout),
        });
        let pl_summarize_y = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("summarization Y"),
            module: &module,
            entry_point: "summarize_y",
            layout: Some(&pl_layout),
        });
        let pl_summarize = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("summarization final"),
            module: &module,
            entry_point: "summarize",
            layout: Some(&pl_layout),
        });
        let pl_normalize = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("value normalization"),
            module: &module,
            entry_point: "normalize",
            layout: Some(&pl_layout),
        });

        let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pl_group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buf_params.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(buf_heightmap.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(buf_gradients.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(buf_errors.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(buf_summary.as_entire_buffer_binding()),
                },
            ],
        });

        // copy heightmap and gradients into GPU buffers
        context.queue.write_buffer(&buf_heightmap, 0, bytemuck::cast_slice(map.data.as_slice()));
        context.queue.write_buffer(&buf_gradients, 0, bytemuck::cast_slice(gradients.data.as_slice()));

        // copy compute parameters
        let k_params = ParamsKernel {
            width: map.size.0 as u32,
            height: map.size.1 as u32,
            step: params.step,
            sqrt2: std::f32::consts::SQRT_2,
        };
        context.queue.write_buffer(&buf_params, 0, bytemuck::bytes_of(&k_params));

        let grid_dim = {
            // compute grid dimensions - divide grid into elements of WG_SIZE by WG_SIZE
            let mut wg_x = map.size.0 / WG_SIZE;
            let mut wg_y = map.size.1 / WG_SIZE;
            if map.size.0 % WG_SIZE != 0 {
                wg_x += 1;
            }
            if map.size.1 % WG_SIZE != 0 {
                wg_y += 1;
            }

            (wg_x as u32, wg_y as u32)
        };

        Ok(Self {
            params, gradients, heightmap: map,
            context,
            buf_params, buf_gradients, buf_heightmap, buf_errors, buf_summary, buf_summary_map,
            bind_group,
            pl_error, pl_diffuse, pl_summarize_y, pl_summarize, pl_normalize,
            grid_dim,
            iters: 0,
        })
    }

    /// Construct an optimizer with an empty heightmap
    pub fn new(params: crate::OptimizeParams, gradients: &'i VectorField<2>) -> Result<Self> {
        let heightmap = Field::new(gradients.size, 0.0);
        Self::new_with_map(params, gradients, heightmap)
    }

    /// Step the optimizer forwards by one iteration
    ///
    /// Returns the accumulated error metric over this iteration.
    pub fn step(&mut self) -> f32 {
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute errors"),
            });
            pass.set_pipeline(&self.pl_error);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.grid_dim.0, self.grid_dim.1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("energy diffusion"),
            });
            pass.set_pipeline(&self.pl_diffuse);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.grid_dim.0, self.grid_dim.1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Y-summarize"),
            });
            pass.set_pipeline(&self.pl_summarize_y);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.grid_dim.0, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("summarize"),
            });
            pass.set_pipeline(&self.pl_summarize);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normalization"),
            });
            pass.set_pipeline(&self.pl_normalize);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.grid_dim.0, self.grid_dim.1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.buf_summary, 0,
            &self.buf_summary_map, 0,
            self.buf_summary.size()
        );

        let cmds = encoder.finish();
        let submission = self.context.queue.submit([cmds]);
        self.context.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        self.iters += 1;

        // retrieve summary
        {
            let buf_slice = self.buf_summary_map.slice(..);
            let is_mapped = std::sync::Arc::new(std::sync::Mutex::new(None));
            let is_mapped_inner = std::sync::Arc::clone(&is_mapped);
            buf_slice.map_async(wgpu::MapMode::Read, move |res| {
                *is_mapped_inner.lock().unwrap() = Some(res);
            });

            while !self.context.device.poll(wgpu::Maintain::Poll) {
            }

            let map_ok = std::sync::Arc::try_unwrap(is_mapped).unwrap()
                        .into_inner().unwrap()
                        .expect("Failed to map output buffer");
            map_ok.expect("Failed to map output buffer");

            let output = {
                let data = buf_slice.get_mapped_range();
                let summ: &Summary = bytemuck::from_bytes(&data);
                summ.total_error
            };

            self.buf_summary_map.unmap();
            output
        }
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn finish(self) -> Field<f32> {
        let out_buf = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            size: self.buf_heightmap.size(),
            mapped_at_creation: false,
        });
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            &self.buf_heightmap, 0,
            &out_buf, 0,
            self.buf_heightmap.size()
        );
        let cmds = encoder.finish();
        let submission = self.context.queue.submit([cmds]);
        self.context.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        let buf_slice = out_buf.slice(..);
        let is_mapped = std::sync::Arc::new(std::sync::Mutex::new(None));
        let is_mapped_inner = std::sync::Arc::clone(&is_mapped);
        buf_slice.map_async(wgpu::MapMode::Read, move |res| {
            *is_mapped_inner.lock().unwrap() = Some(res);
        });

        while !self.context.device.poll(wgpu::Maintain::Poll) {
        }

        let map_ok = std::sync::Arc::try_unwrap(is_mapped).unwrap()
                    .into_inner().unwrap()
                    .expect("Failed to map output buffer");
        map_ok.expect("Failed to map output buffer");

        let data = buf_slice.get_mapped_range();
        let data_structured: &[f32] = bytemuck::cast_slice(&data);

        assert_eq!(data_structured.len(), self.heightmap.data.len());
        let mut hmap = self.heightmap;
        hmap.data.copy_from_slice(data_structured);

        hmap
    }
}
