// shader.slang

static float2 positions[3] = float2[](
    float2(0.0, -0.5),
    float2(0.5,  0.5),
    float2(-0.5, 0.5)
);

static float3 colors[3] = float3[](
    float3(1.0, 0.0, 0.0), // red
    float3(0.0, 1.0, 0.0), // green
    float3(0.0, 0.0, 1.0)  // blue
);

struct VertexOutput {
    float3 color;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(uint vid : SV_VertexID)
{
    VertexOutput outVal;
    outVal.sv_position = float4(positions[vid], 0.0, 1.0);
    outVal.color       = colors[vid];
    return outVal;
}

[shader("fragment")]
float4 fragColor(VertexOutput inVert) : SV_Target
{
    return float4(inVert.color, 1.0);
}

[shader("fragment")]
float4 fragGray(VertexOutput inVert) : SV_Target
{
    float g = dot(inVert.color, float3(0.299, 0.587, 0.114));
    return float4(g, g, g, 1.0);
}

