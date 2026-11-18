struct Vertex {
    float3 position;
    float3 color;
    float2 tex_coord;
};

struct SceneInfo {
    float2 resolution;
    float anim_value;
    uint num_data;
    uint tex_id;
    uint3 _pad0;
};

struct PushData {
    uint vertex_buffer;
    float rotation_angle;
    uint num_vertex;
    uint _pad0;
    uint scene_info;
    uint points;
    uint texture_image;
    uint texture_sampler;
    float4 color;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float2> point_buffers[];

[[vk::binding(1, 0)]]
RWStructuredBuffer<Vertex> vertex_buffers[];

[[vk::binding(2, 0)]]
Texture2D<float4> sampled_images[];

[[vk::binding(4, 0)]]
SamplerState samplers[];

[[vk::binding(5, 0)]]
ConstantBuffer<SceneInfo> scene_infos[];

[[vk::push_constant]]
cbuffer PC {
    PushData pc;
};

struct VertexInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR0;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    if (index >= pc.num_vertex) {
        return;
    }

    RWStructuredBuffer<Vertex> vertices = vertex_buffers[pc.vertex_buffer];
    Vertex vertex = vertices[index];

    float angle = pc.rotation_angle * sin((float(index) + 1.0) * 2.3);
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);

    float3 position = vertex.position;
    // Rotated by -angle, matching the reference sample's GLSL matrix (its
    // `mat3(...)` constructor is column-major, so the same literals rotate the
    // other way than a row-major `float3x3` would).
    vertex.position = float3(
        cos_angle * position.x + sin_angle * position.y,
        -sin_angle * position.x + cos_angle * position.y,
        position.z
    );

    vertices[index] = vertex;
}

[shader("vertex")]
VertexOutput vertexMain(VertexInput input) {
    SceneInfo scene = scene_infos[pc.scene_info];
    float aspect_ratio = scene.resolution.y / max(scene.resolution.x, 1.0);

    float3 position = input.position;
    position.x *= aspect_ratio;

    VertexOutput output;
    output.position = float4(position, 1.0);
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

float4 shade_fragment(VertexOutput input, bool use_texture) {
    SceneInfo scene = scene_infos[pc.scene_info];
    float2 resolution = max(scene.resolution, float2(1.0, 1.0));
    float2 frag_pos = (input.position.xy / resolution) * 2.0 - 1.0;

    StructuredBuffer<float2> points = point_buffers[pc.points];
    float min_dist = 1.0e10;
    for (uint i = 0; i < scene.num_data; i++) {
        min_dist = min(min_dist, distance(frag_pos, points[i]));
    }

    float radius = 0.02;
    float edge_smooth = 0.01;
    float alpha = 1.0 - smoothstep(radius, radius - edge_smooth, min_dist);

    float4 point_color = float4(scene.anim_value * pc.color.rgb, 1.0);
    float4 triangle_color = float4(input.color, 1.0);

    if (use_texture) {
        triangle_color *= sampled_images[pc.texture_image].Sample(
            samplers[pc.texture_sampler],
            input.uv
        );
    }

    return lerp(point_color, triangle_color, alpha);
}

[shader("fragment")]
float4 fragmentMainColor(VertexOutput input) : SV_Target0 {
    return shade_fragment(input, false);
}

[shader("fragment")]
float4 fragmentMainTextured(VertexOutput input) : SV_Target0 {
    return shade_fragment(input, true);
}
