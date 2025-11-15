// C = A * B
// A: M x K, B: K x N, C: M x N
// Flattened row-major buffers.

static const uint TILE_SIZE = 16;

struct MatMulParams
{
    uint M;  // rows of A and C
    uint N;  // cols of B and C
    uint K;  // cols of A / rows of B
};

// Mark this as Vulkan push constants
[[vk::push_constant]]
MatMulParams gParams;

// Storage buffers for A, B, C
// binding set = 0 by default (you can change if needed)
[[vk::binding(0)]]
StructuredBuffer<float> bufferA;

[[vk::binding(1)]]
StructuredBuffer<float> bufferB;

[[vk::binding(2)]]
RWStructuredBuffer<float> bufferC;

// Shared memory tiles (one tile of A and one tile of B per workgroup)
groupshared float tileA[TILE_SIZE][TILE_SIZE];
groupshared float tileB[TILE_SIZE][TILE_SIZE];

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(
    uint3 groupID        : SV_GroupID,
    uint3 groupThreadID  : SV_GroupThreadID,
    uint3 dispatchThreadID : SV_DispatchThreadID
)
{
    uint M = gParams.M;
    uint N = gParams.N;
    uint K = gParams.K;

    // Global coordinates in C
    uint row = dispatchThreadID.y;
    uint col = dispatchThreadID.x;

    // Out-of-bounds threads do nothing
    if (row >= M || col >= N)
        return;

    float acc = 0.0f;

    // Number of tiles along the K dimension
    uint numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (uint tileIndex = 0; tileIndex < numTiles; ++tileIndex)
    {
        // Index within current tile for loads
        uint localRow = groupThreadID.y;
        uint localCol = groupThreadID.x;

        // Global K indices for this tile
        uint kA = tileIndex * TILE_SIZE + localCol; // column in A
        uint kB = tileIndex * TILE_SIZE + localRow; // row in B

        // Load A tile: element A[row, kA]
        if (row < M && kA < K)
        {
            tileA[localRow][localCol] = bufferA[row * K + kA];
        }
        else
        {
            tileA[localRow][localCol] = 0.0f;
        }

        // Load B tile: element B[kB, col]
        if (kB < K && col < N)
        {
            tileB[localRow][localCol] = bufferB[kB * N + col];
        }
        else
        {
            tileB[localRow][localCol] = 0.0f;
        }

        // Make sure all threads see the tile data
        GroupMemoryBarrierWithGroupSync();

        // Compute partial dot product for this tile
        [unroll]
        for (uint k = 0; k < TILE_SIZE; ++k)
        {
            acc += tileA[localRow][k] * tileB[k][localCol];
        }

        // Sync before loading the next tile
        GroupMemoryBarrierWithGroupSync();
    }

    // Write out result C[row, col]
    bufferC[row * N + col] = acc;
}

