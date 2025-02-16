#define TILE_WIDTH 16

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

//@@ Tune your tile size
#define TILE_WIDTH 16
#define MAX_KERNEL_SZ 7

__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __local float X_shared[TILE_WIDTH + MAX_KERNEL_SZ - 1][TILE_WIDTH + MAX_KERNEL_SZ - 1];

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int b = get_global_id(2);
    int m = get_group_id(0);

    // Determine the position of the element in the output
    int h = (get_group_id(1) / W_grid) * TILE_WIDTH + get_local_id(1);
    int w = (get_group_id(1) % W_grid) * TILE_WIDTH + get_local_id(0);

    float accum = 0.0f;

    // Iterate over all channels
    for (int c = 0; c < C; c++) {
        // Load data into shared memory
        X_shared[get_local_id(1)][get_local_id(0)] = x4d(b, c, h, w);

        // Load bottom elements (halo)
        if (get_local_id(1) < K - 1 && h + TILE_WIDTH < H) {
            X_shared[get_local_id(1) + TILE_WIDTH][get_local_id(0)] = x4d(b, c, h + TILE_WIDTH, w);
        }

        // Load right elements (halo)
        if (get_local_id(0) < K - 1 && w + TILE_WIDTH < W) {
            X_shared[get_local_id(1)][get_local_id(0) + TILE_WIDTH] = x4d(b, c, h, w + TILE_WIDTH);
        }

        // Load bottom right elements (halo)
        if (get_local_id(1) > TILE_WIDTH - K && get_local_id(0) > TILE_WIDTH - K && h + K - 1 < H && w + K - 1 < W) {
            X_shared[get_local_id(1) + K - 1][get_local_id(0) + K - 1] = x4d(b, c, h + K - 1, w + K - 1);
        }

	// Synchronize to make sure all elements have been loaded into shared memory
        barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the output convolution using kernel over data in the shared memory
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                accum += X_shared[get_local_id(1) + p][get_local_id(0) + q] * k4d(m, c, p, q);
            }
        }

	// Synchronize before proceeding to the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // If in bounds, write the accumulated restult to the output
    if (h < H_out && w < W_out) {
        y4d(b, m, h, w) = accum;
    }

#undef y4d
#undef x4d
#undef k4d
}