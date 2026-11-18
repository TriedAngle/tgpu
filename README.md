# TryAngle's GPU Library

Easy to use multi purpose compute and graphics library

## Examples
run with `cargo run --bin <name>` in examples/

## C API
`tgpu-native` builds a `cdylib`/`staticlib` plus and `tgpu-native/include/tgpu.h`.

```sh
cd tgpu-native/examples
make
./particles
./minimal
```

### Regenerating the C header
```sh
cargo install cbindgen   # once
cd tgpu-native
cbindgen --config cbindgen.toml --output include/tgpu.h
```