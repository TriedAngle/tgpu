// C = A * B
// A: M x K, row-major
// B^T: N x K, row-major (B transposed on CPU)
// C: M x N, row-major
// Assumptions:
//  - M, N, K are multiples of 4
//  - K is a multiple of TILE_SIZE (16) for clean tiling
//  - A and B^T are uploaded as plain float arrays, but bound as float4

static const uint TILE_SIZE = 16;

struct MatMulParams {
    uint M;  // rows of A and C
    uint N;  // cols of B and C
    uint K;  // cols of A / rows of B (multiple of 16)
};

[[vk::push_constant]]
MatMulParams gParams;

[[vk::binding(0)]]
StructuredBuffer<float4> bufferA;   // A: M x (K/4) in float4

[[vk::binding(1)]]
StructuredBuffer<float4> bufferB;   // B^T: N x (K/4) in float4

[[vk::binding(2)]]
RWStructuredBuffer<float> bufferC;  // C: M x N, row-major floats

groupshared float4 tileA[TILE_SIZE][TILE_SIZE / 4]; // 16 x 4
groupshared float4 tileB[TILE_SIZE][TILE_SIZE / 4]; // 16 x 4

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(
    uint3 groupID           : SV_GroupID,
    uint3 groupThreadID     : SV_GroupThreadID,
    uint3 dispatchThreadID  : SV_DispatchThreadID
) {
    uint M = gParams.M;
    uint N = gParams.N;
    uint K = gParams.K;

    uint row = dispatchThreadID.y;  // C row
    uint col = dispatchThreadID.x;  // C col

    if (row >= M || col >= N) {
        return;
    }

    const uint K4       = K / 4;            // K in float4s
    const uint TILE_K4  = TILE_SIZE / 4;    // 16 / 4 = 4
    const uint numTiles = K4 / TILE_K4;     // requires K % 16 == 0

    uint localRow = groupThreadID.y; // 0..15
    uint localCol = groupThreadID.x; // 0..15

    float acc = 0.0f;

    for (uint tileIndex = 0; tileIndex < numTiles; ++tileIndex) {
        uint k4Base = tileIndex * TILE_K4;

        if (localCol < TILE_K4) {
            uint aIndex4 = row * K4 + (k4Base + localCol);
            tileA[localRow][localCol] = bufferA[aIndex4];
        }

        if (localRow < TILE_K4) {
            uint bIndex4 = col * K4 + (k4Base + localRow);
            tileB[localCol][localRow] = bufferB[bIndex4];
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint j = 0; j < TILE_K4; ++j) {
            float4 aVec = tileA[localRow][j];
            float4 bVec = tileB[localCol][j];
            acc += dot(aVec, bVec);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    bufferC[row * N + col] = acc;
}

