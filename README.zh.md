# TryAngle's GPU Library

易于使用的多功能计算与图形库

## 示例
在 examples/ 下运行 `cargo run --bin <name>`

## C API
`tgpu-native` 生成 `cdylib`/`staticlib` 以及 `tgpu-native/include/tgpu.h`。

```sh
cd tgpu-native/examples
make
./particles
./minimal
```

### 重新生成 C 头文件
```sh
cargo install cbindgen   # 只需一次
cd tgpu-native
cbindgen --config cbindgen.toml --output include/tgpu.h
```