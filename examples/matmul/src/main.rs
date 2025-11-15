use rand::Rng;
use tgpu::ash::vk;

const MATMUL_WGSL: &str = r#"
struct Push {
    m: u32, // rows of A and C
    n: u32, // cols of B and C
    k: u32, // cols of A / rows of B
}

@group(0) @binding(0)
var<storage, read>  A: array<f32>;
@group(0) @binding(1)
var<storage, read>  B: array<f32>;
@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

var<push_constant> pc: Push;

fn idx_a(r: u32, c: u32) -> u32 { return r * pc.k + c; }
fn idx_b(r: u32, c: u32) -> u32 { return r * pc.n + c; }
fn idx_c(r: u32, c: u32) -> u32 { return r * pc.n + c; }

const TILE: u32 = 16u;

// Flattened 1D tiles -> avoids nested-array layout issues
var<workgroup> tileA: array<f32, 512>;
var<workgroup> tileB: array<f32, 512>;

// 2D -> 1D index helper for tiles
fn tix(x: u32, y: u32) -> u32 {
    return y * TILE + x;
}

@compute @workgroup_size(TILE, TILE, 1)
fn main(@builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>) {

    let row = gid.y; // C row
    let col = gid.x; // C col

    if (row >= pc.m || col >= pc.n) {
        return;
    }

    var acc: f32 = 0.0;
    let numTiles = (pc.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE + lid.x;
        let bRow = t * TILE + lid.y;

        // Load A tile (row, aCol)
        if (aCol < pc.k) {
            tileA[tix(lid.x, lid.y)] = A[idx_a(row, aCol)];
        } else {
            tileA[tix(lid.x, lid.y)] = 0.0;
        }

        // Load B tile (bRow, col)
        if (bRow < pc.k) {
            tileB[tix(lid.x, lid.y)] = B[idx_b(bRow, col)];
        } else {
            tileB[tix(lid.x, lid.y)] = 0.0;
        }

        workgroupBarrier();

        // Accumulate for this tile
        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            // tileA[y][i] * tileB[i][x] -> flattened:
            acc = acc + tileA[tix(i,      lid.y)] * tileB[tix(lid.x, i)];
        }

        workgroupBarrier();
    }

    C[idx_c(row, col)] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    m: u32,
    n: u32,
    k: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_module("naga", log::LevelFilter::Warn)
        .init();

    // Problem sizes
    let (m, n, k) = (512u32, 512u32, 512u32);
    let len_a = (m * k) as usize;
    let len_b = (k * n) as usize;
    let len_c = (m * n) as usize;

    // Generate inputs
    let mut rng = rand::rng();
    let mut host_a = vec![0.0f32; len_a];
    let mut host_b = vec![0.0f32; len_b];
    for x in &mut host_a {
        *x = rng.random_range(-1.0..1.0);
    }
    for x in &mut host_b {
        *x = rng.random_range(-1.0..1.0);
    }

    let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
        app_name: "Headless MatMul",
        engine_name: "Example Engine",
    })?;

    let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
    let adapter = adapters[0].clone();

    let (device, mut queues) = instance.request_device(
        &tgpu::DeviceCreateInfo {},
        adapter,
        &[tgpu::QueueRequest {
            required_flags: tgpu::QueueFlags::COMPUTE | tgpu::QueueFlags::TRANSFER,
            exclude_flags: tgpu::QueueFlags::empty(),
            strict: false,
            allow_fallback_share: true,
        }],
    )?;
    let queue = queues.next().unwrap();

    let usage_rw_host = tgpu::BufferUsage::STORAGE
        | tgpu::BufferUsage::COPY_DST
        | tgpu::BufferUsage::COPY_SRC
        | tgpu::BufferUsage::DEVICE
        | tgpu::BufferUsage::COHERENT
        | tgpu::BufferUsage::HOST_VISIBLE
        | tgpu::BufferUsage::MAP_WRITE
        | tgpu::BufferUsage::MAP_READ;

    let buf_a = device.create_buffer(&tgpu::BufferInfo {
        label: Some(tgpu::Label::Name("A")),
        size: std::mem::size_of::<f32>() * len_a,
        usage: usage_rw_host,
    })?;
    let buf_b = device.create_buffer(&tgpu::BufferInfo {
        label: Some(tgpu::Label::Name("B")),
        size: std::mem::size_of::<f32>() * len_b,
        usage: usage_rw_host,
    })?;
    let buf_c = device.create_buffer(&tgpu::BufferInfo {
        label: Some(tgpu::Label::Name("C")),
        size: std::mem::size_of::<f32>() * len_c,
        usage: usage_rw_host,
    })?;

    buf_a.write(bytemuck::cast_slice(&host_a), 0);
    buf_b.write(bytemuck::cast_slice(&host_b), 0);

    let dsl = device.create_descriptor_set_layout(&tgpu::DescriptorSetLayoutInfo {
        label: Some(tgpu::Label::Name("MatMul DSL")),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        bindings: &[
            tgpu::DescriptorBinding::unique(
                0,
                tgpu::DescriptorType::StorageBuffer,
                tgpu::ShaderStageFlags::COMPUTE,
            ),
            tgpu::DescriptorBinding::unique(
                1,
                tgpu::DescriptorType::StorageBuffer,
                tgpu::ShaderStageFlags::COMPUTE,
            ),
            tgpu::DescriptorBinding::unique(
                2,
                tgpu::DescriptorType::StorageBuffer,
                tgpu::ShaderStageFlags::COMPUTE,
            ),
        ],
    });

    let pool = device.create_descriptor_pool(&tgpu::DescriptorPoolInfo {
        label: Some(tgpu::Label::Name("MatMul Pool")),
        max_sets: 1,
        layouts: &[&dsl],
        flags: vk::DescriptorPoolCreateFlags::empty(),
    });

    let dset = device.create_descriptor_set(pool.clone(), &dsl);
    dset.write(&[
        tgpu::DescriptorWrite::StorageBuffer {
            binding: 0,
            buffer: &buf_a,
            offset: 0,
            range: vk::WHOLE_SIZE,
            array_element: None,
        },
        tgpu::DescriptorWrite::StorageBuffer {
            binding: 1,
            buffer: &buf_b,
            offset: 0,
            range: vk::WHOLE_SIZE,
            array_element: None,
        },
        tgpu::DescriptorWrite::StorageBuffer {
            binding: 2,
            buffer: &buf_c,
            offset: 0,
            range: vk::WHOLE_SIZE,
            array_element: None,
        },
    ]);

    let shader = device
        .create_shader(None, &tgpu::ShaderSource::Wgsl(MATMUL_WGSL))
        .expect("MatMul WGSL");
    let pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
        label: Some(tgpu::Label::Name("MatMul Pipeline")),
        shader: shader.entry("main"),
        push_constant_size: Some(std::mem::size_of::<Push>() as u32),
        descriptor_layouts: &[&dsl],
        cache: None,
    });

    let tile: u32 = 16;
    let groups_x = (n + tile - 1) / tile;
    let groups_y = (m + tile - 1) / tile;
    let push = Push { m, n, k };

    let mut rec = queue.record();
    rec.bind_compute_pipeline(&pipeline);
    rec.bind_compute_descriptor_set(&dset, &pipeline, 0, &[]);
    rec.push_compute_constants(&pipeline, push);
    rec.dispatch(groups_x, groups_y, 1);

    queue.submit(tgpu::SubmitInfo {
        records: &[rec.finish()],
        ..Default::default()
    });

    device.wait_idle();

    let mut host_c = vec![0.0f32; len_c];
    let bytes_c = std::mem::size_of::<f32>() * len_c;
    buf_c.read(bytemuck::cast_slice_mut(&mut host_c), 0, bytes_c);

    // Tiny correctness spot-check against CPU on a few random entries
    let mut max_abs_err = 0.0f32;
    for _ in 0..10 {
        let r = rng.random_range(0..m as usize);
        let c = rng.random_range(0..n as usize);
        let mut cpu = 0.0f32;
        for kk in 0..k as usize {
            cpu += host_a[r * k as usize + kk] * host_b[kk * n as usize + c];
        }
        let gpu = host_c[r * n as usize + c];
        max_abs_err = max_abs_err.max((cpu - gpu).abs());
    }

    println!(
        "MatMul {}x{} * {}x{} = {}x{} done. Max abs error (10 samples): {:.3e}",
        m, k, k, n, m, n, max_abs_err
    );

    Ok(())
}
