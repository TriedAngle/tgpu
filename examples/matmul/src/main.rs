use rand::Rng;

const SHADER: &str = include_str!("./shader.slang");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    a: tgpu::ReadBufferHandle,
    b: tgpu::ReadBufferHandle,
    c: tgpu::RwBufferHandle,
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

    // transposing for row major
    let mut host_b_t = vec![0.0f32; len_b];
    for row in 0..(k as usize) {
        for col in 0..(n as usize) {
            host_b_t[col * (k as usize) + row] = host_b[row * (n as usize) + col];
        }
    }

    let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
        app_name: "Headless MatMul",
        engine_name: "Example Engine",
    })?;

    let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
    let adapter = adapters[0].clone();

    let (device, mut queues) = instance.request_device(
        &tgpu::DeviceCreateInfo::default(),
        adapter,
        &[tgpu::QueueRequest {
            required_flags: tgpu::QueueFlags::COMPUTE | tgpu::QueueFlags::TRANSFER,
            exclude_flags: tgpu::QueueFlags::empty(),
            strict: false,
            allow_fallback_share: true,
        }],
    )?;
    let queue = queues.next().unwrap();

    let storage_buffer = tgpu::BufferDesc {
        usage: tgpu::BufferUses::STORAGE,
        memory: tgpu::MemoryPreset::Dynamic,
        host_access: tgpu::HostAccess::ReadWriteRandom,
        ..Default::default()
    };

    let buf_a = device.create_buffer(&tgpu::BufferDesc {
        label: Some(tgpu::Label::Name("A")),
        size: std::mem::size_of::<f32>() * len_a,
        ..storage_buffer.clone()
    })?;
    let buf_b = device.create_buffer(&tgpu::BufferDesc {
        label: Some(tgpu::Label::Name("B")),
        size: std::mem::size_of::<f32>() * len_b,
        ..storage_buffer.clone()
    })?;
    let buf_c = device.create_buffer(&tgpu::BufferDesc {
        label: Some(tgpu::Label::Name("C")),
        size: std::mem::size_of::<f32>() * len_c,
        ..storage_buffer
    })?;

    buf_a.write_slice(&host_a);
    buf_b.write_slice(&host_b_t);

    let bindless = device.create_bindless_heap(&tgpu::BindlessInfo {
        max_read_buffers: 2,
        max_rw_buffers: 1,
        ..Default::default()
    });

    let buf_a_handle = bindless.add_read_buffer(&buf_a);
    let buf_b_handle = bindless.add_read_buffer(&buf_b);
    let buf_c_handle = bindless.add_rw_buffer(&buf_c);

    let shader = device
        .create_shader(None, tgpu::ShaderSource::Slang(SHADER.as_bytes()))
        .expect("MatMul Slang");

    let pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
        label: Some(tgpu::Label::Name("MatMul Pipeline")),
        shader: shader.entry("main"),
        push_constant_size: Some(std::mem::size_of::<Push>() as u32),
        descriptor_layouts: &[bindless.layout()],
        cache: None,
    });

    let tile: u32 = 16;
    let groups_x = n.div_ceil(tile);
    let groups_y = m.div_ceil(tile);
    let push = Push {
        a: buf_a_handle,
        b: buf_b_handle,
        c: buf_c_handle,
        m,
        n,
        k,
    };

    let mut rec = queue.record();
    rec.bind_compute_pipeline(&pipeline);
    rec.bind_compute_descriptor_set(bindless.descriptor_set(), &pipeline, 0, &[]);
    rec.push_compute_constants(&pipeline, push);
    rec.dispatch(groups_x, groups_y, 1);

    queue.submit(tgpu::SubmitInfo {
        records: &[rec.finish()],
        ..Default::default()
    });

    device.wait_idle();

    let mut host_c = vec![0.0f32; len_c];
    buf_c.read_slice(&mut host_c);

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
