struct ComputeParams {
    width: u32,
    height: u32,
    step: f32,
    sqrt2: f32,
}

struct Summary {
    min_h: f32,
    max_h: f32,
    error: f32,
}

@group(0) @binding(0)
var<uniform> params: ComputeParams;

@group(0) @binding(1)
var<storage, read_write> height : array<f32>;

@group(0) @binding(2)
var<storage> gradients : array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> errors : array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> summary : Summary;

fn idx_for(x: u32, y: u32) -> u32 {
    return x + y*params.width;
}

@compute
@workgroup_size(8, 8)
fn compute_errors(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;
    let sqrt2 = params.sqrt2;

    if(x >= params.width || y >= params.height) {
        return;
    }

    let idx = idx_for(x, y);
    let cur = height[idx];

    // compute gradient for this cell
    let up_y    = select(params.height - 1u, y - 1u, y > 0u);
    let down_y  = select(0u, y + 1u, (y + 1u) < params.height);
    let left_x  = select(params.width - 1u, x - 1u, x > 0u);
    let right_x = select(0u, x + 1u, (x + 1u) < params.width);

    var dy: f32 = 0.0;
    var dx: f32 = 0.0;

    // up-left
    {
        let v = height[idx_for(left_x, up_y)];
        dx += (cur - v) / sqrt2;
        dy += (cur - v) / sqrt2;
    }
    dy += cur - height[idx_for(x, up_y)]; // up
    // up-right
    {
        let v = height[idx_for(right_x, up_y)];
        dx += (v - cur) / sqrt2;
        dy += (cur - v) / sqrt2;
    }
    dx += cur - height[idx_for(left_x, y)]; // left
    dx += height[idx_for(right_x, y)] - cur; // right

    // down-left
    {
        let v = height[idx_for(left_x, down_y)];
        dx += (v - cur) / sqrt2;
        dy += (cur - v) / sqrt2;
    }
    dy += height[idx_for(x, down_y)] - cur; // down
    // down-right
    {
        let v = height[idx_for(right_x, down_y)];
        dx += (v - cur) / sqrt2;
        dy += (v - cur) / sqrt2;
    }

    let grad = vec2<f32>(dx / 6.0, dy / 6.0);
    errors[idx] = gradients[idx] - grad;
}

@compute
@workgroup_size(8, 8)
fn energy_diffusion(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;
    let sqrt2 = params.sqrt2;
    let step = params.step;

    if(x >= params.width || y >= params.height) {
        return;
    }

    let idx = idx_for(x, y);
    let err = errors[idx];
    var item = height[idx];

    let dot_ul = err.x/-sqrt2 + err.y/-sqrt2;
    let dot_u = -err.y;
    let dot_ur = err.x/sqrt2 + err.y/-sqrt2;
    let dot_l = -err.x;
    let dot_r = err.x;
    let dot_dl = err.x/-sqrt2 + err.y/sqrt2;
    let dot_d = err.y;
    let dot_dr = err.x/sqrt2 + err.y/sqrt2;

    let has_l = x > 0u;
    let has_r = (x + 1u) < params.width;
    let has_u = y > 0u;
    let has_d = (y + 1u) < params.height;

    // energy xfer up-left
    if(has_l && has_u) {
        let ul = errors[idx_for(x - 1u, y - 1u)];
        item -= dot_ul * step; // to up-left
        item += ((ul.x / sqrt2) + (ul.y / sqrt2)) * step; // from up-left (going down-right)
    }

    // energy xfer up
    if(has_u) {
        item -= dot_u * step; // out of self (i.e. to up)
        item += errors[idx_for(x, y - 1u)].y * step; // from up (going down)
    }

    // energy xfer up-right
    if(has_r && has_u) {
        let ur = errors[idx_for(x + 1u, y - 1u)];
        item -= dot_ur * step; // to up-left
        item += ((ur.x / -sqrt2) + (ur.y / sqrt2)) * step; // from up-right (going down-left)
    }

    // energy xfer left
    if(has_l) {
        item -= dot_l * step; // out of self
        item += errors[idx_for(x - 1u, y)].x * step; // from left
    }

    // energy xfer right
    if(has_r) {
        item -= dot_r * step; // out of self
        item += -errors[idx_for(x + 1u, y)].x * step; // from right
    }

    // energy xfer down-left
    if(has_l && has_d) {
        let dl = errors[idx_for(x - 1u, y + 1u)];
        item -= dot_dl * step; // to down-left
        item += ((dl.x / sqrt2) + (dl.y / -sqrt2)) * step; // from down-left (going up-right)
    }

    // energy xfer down
    if(has_d) {
        item -= dot_d * step; // out of self
        item += -errors[idx_for(x, y + 1u)].y * step; // from down
    }

    // energy xfer down-right
    if(has_l && has_d) {
        let dr = errors[idx_for(x + 1u, y + 1u)];
        item -= dot_dr * step; // to down-right
        item += ((dr.x / -sqrt2) + (dr.y / -sqrt2)) * step; // from down-right (going up-left)
    }

    height[idx] = item;
}

@compute
@workgroup_size(8, 1)
fn summarize_y(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;

    if(x >= params.width || y != 0u) {
        return;
    }

    // compute maximum and minimum
    //
    // reuse the error vectors as temporary storage here

    // top-to-bottom reduction
    let start_idx = idx_for(x, 0u);
    let a = height[idx_for(x, 0u)];
    var out = vec2<f32>(a, a);
    var accum = 0.0;

    for(var i: u32 = 0u; i < params.height;i++) {
        let idx = idx_for(x, i);

        // re-derive initial gradient vector
        let error = errors[idx];
        let goal = gradients[idx];
        let current = -(error - goal);
        accum += abs(1.0 - dot(current, goal));

        // compute min/max values
        let val = height[idx];
        out.x = min(out.x, val);
        out.y = max(out.y, val);
    }

    errors[idx_for(x, 0u)] = out;
    errors[idx_for(x, 1u)] = vec2<f32>(accum, accum);
}

@compute
@workgroup_size(1, 1)
fn summarize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;

    if(x != 0u || y != 0u) {
        return;
    }

    // final reduction
    var out = errors[0];
    var accum = 0.0;

    for(var i: u32 = 0u; i < params.width;i++) {
        let val = errors[i];
        out.x = min(out.x, val.x);
        out.y = max(out.y, val.y);
        accum += errors[i + params.width].x;
    }

    summary.min_h = out.x;
    summary.max_h = out.y;
    summary.error = accum;
}

@compute
@workgroup_size(8, 8)
fn normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;

    if(x >= params.width || y >= params.height) {
        return;
    }

    // mapping
    let me = height[idx_for(x, y)];
    height[idx_for(x, y)] = (me - summary.min_h) / max((summary.max_h - summary.min_h), 1.0);
}
