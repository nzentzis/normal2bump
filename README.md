This is a tool for converting tangent-space normal maps into bump/heightmaps. I
originally wrote it while trying to bake a game model's normal map into a mesh
for 3D printing, but it's not specific to that use-case.

# Usage
Invoke the compiled executable, passing your normal-map as the first argument.
You can optionally specify an output path as the second argument. It'll chew on
it for a while, and eventually write out a heightmap to the output path.

If you have a GPU capable of running compute shaders, the tool *should* use a
GPU implementation of the algorithm and the run shouldn't take more than a few
seconds. If your GPU can't support that or if the GPU optimizer can't be
initialized for some reason, the tool will fall back to a CPU-based optimizer.
The CPU optimizer will take advantage of all your cores, but it's still a lot
slower than the GPU one.

# How it Works
The tool first interprets the input normal-map as a grid of 3D tangent-space
vectors, and takes the X and Y components of each one. The resulting grid of 2D
vectors is referred to as the *target field*, and represents the desired
gradient of the final height-map at each point. This ignores mesh topology and
shape, but that limitation hasn't been a problem for my usage.

Once the target field is known, the heightmap is initialized to all zeros
(floating-point, so it can go below zero). The tool then repeatedly makes small
adjustments to the heightmap to change its gradient field to more closely match
the target.

Each iteration consists of the following steps:

 1. Compute the gradient field of the current heightmap. The resulting field of
    2D vectors is of the same size as the heightmap, and each cell stores the
    gradient (partial derivative with respect to X and Y) of the heightmap at
    that point.
 2. Compute the error, which is a vector field defined by `target - grad(heightmap)`
    at each cell. This step also computes a "total error" metric, defined as the
    sum of `abs(1 - dot(target, grad(heightmap)))` over all cells.
 3. Update the heightmap to reduce the error. For each cell, this transfers (i.e.
    reduces the cell's height and increases another cell's height, or vice-versa)
    height between a cell and its neighbors, to an extent defined by the product
    of the step size and `dot(error, neighbor_pos - cell_pos)`.
 4. Normalize the heightmap. Find the minimum and maximum heightmap values, and
    scale all values such that these become 0 and 1, respectively.

Iteration continues until the difference between the largest and smallest "total
error" values during the past 1000 iterations is below a defined threshold (by
default, 1e-10). The heightmap is then quantized into 8-bit integers and saved
to the output path.
