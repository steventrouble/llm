# Contributors Guide

The purpose of this document is to make it easy for open-source community
members to contribute to this project. We'd love to discuss your contributions
with you via a GitHub [Issue](https://github.com/rustformers/llm/issues/new) or
[Discussion](https://github.com/rustformers/llm/discussions/new?category=ideas),
or on [Discord](https://discord.gg/YB9WaXYAWU)!

## Checking Changes

This project uses a [GitHub workflow](../.github/workflows/rust.yml) to enforce
code standards.

The `rusty-hook` project is used to run a similar set of checks automatically before committing.
If you would like to run these checks locally, use `cargo run -p precommit-check`.

## Regenerating GGML Bindings

Follow these steps to update the GGML submodule and regenerate the Rust bindings
(this is only necessary if your changes depend on new GGML features):

```shell
git submodule update --remote
cargo run --release --package generate-ggml-bindings
```

## Acceleration Support for Building

The `ggml-sys` crate includes various acceleration backends, selectable via `--features` flags. The availability of supported backends varies by platform, and `ggml-sys` can only be built with a single active acceleration backend at a time. If cublas and clblast are both specified, cublas is prioritized and clblast is ignored.

| Platform/OS | `cublas`           | `clblast`          | `metal`            |
| ----------- | ------------------ | ------------------ | ------------------ |
| Windows     | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| Linux       | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| MacOS       | :x:                | :x:                | :heavy_check_mark: |

## Dependencies for Building with Acceleration Support

### Windows

#### CuBLAS

CUDA must be installed. You can download CUDA from the official [Nvidia site](https://developer.nvidia.com/cuda-downloads).

#### CLBlast

CLBlast can be installed via [vcpkg](https://vcpkg.io/en/getting-started.html) using the command `vcpkg install clblast`. After installation, the `OPENCL_PATH` and `CLBLAST_PATH` environment variables should be set to the `opencl_x64-windows` and `clblast_x64-windows` directories respectively.

Here's an example of the required commands:

```
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install clblast
set OPENCL_PATH=....\vcpkg\packages\opencl_x64-windows
set CLBLAST_PATH=....\vcpkg\packages\clblast_x64-windows
```

⚠️ When working with MSVC in a Windows environment, it is essential to set the `-Ctarget-feature=+crt-static` Rust flag. This flag is critical as it enables the static linking of the C runtime, which can be paramount for certain deployment scenarios or specific runtime environments.

To set this flag, you can modify the .cargo\config file in your project directory. Please add the following configuration snippet:

```
[target.x86_64-pc-windows-msvc]
rustflags = ["-Ctarget-feature=+crt-static"]
```

This will ensure the Rust flag is appropriately set for your compilation process.

For a comprehensive guide on the usage of Rust flags, including other possible ways to set them, please refer to this detailed [StackOverflow discussion](https://stackoverflow.com/questions/38040327/how-to-pass-rustc-flags-to-cargo). Make sure to choose an option that best fits your project requirements and development environment.

⚠️ For `llm` to function properly, it requires the `clblast.dll` and `OpenCL.dll` files. These files can be found within the `bin` subdirectory of their respective vcpkg packages. There are two options to ensure `llm` can access these files:

1. Amend your `PATH` environment variable to include the `bin` directories of each respective package.

2. Manually copy the `clblast.dll` and `OpenCL.dll` files into the `./target/release` or `./target/debug` directories. The destination directory will depend on the profile that was active during the compilation process.

Please choose the option that best suits your needs and environment configuration.

### Linux

#### CuBLAS

You need to have CUDA installed on your system. CUDA can be downloaded and installed from the official [Nvidia site](https://developer.nvidia.com/cuda-downloads). On Linux distributions that do not have CUDA_PATH set, the environment variables CUDA_INCLUDE_PATH and CUDA_LIB_PATH can be set to their corresponding paths.

#### CLBlast

CLBlast can be installed on Linux through various package managers. For example, using `apt` you can install it via `sudo apt install clblast`. After installation, make sure that the `OPENCL_PATH` and `CLBLAST_PATH` environment variables are correctly set. Additionally the environment variables OPENCL_INCLUDE_PATH/OPENCL_LIB_PATH & CBLAST_INCLUDE_PATH/CLBLAST_LIB_PATH can be used to specify the location of the files. All environment variables are supported by all listed operating systems.

### MacOS

#### Metal

Xcode and the associated command-line tools should be installed on your system, and you should be running a version of MacOS that supports Metal. For more detailed information, please consult the [official Metal documentation](https://developer.apple.com/metal/).

To enable Metal using the CLI, ensure it was built successfully using `--features=metal` and then pass the `--use-gpu` flag.

The current underlying implementation of Metal in GGML is still in flux and has some limitations:

- Metal for GGML requires the `ggml-metal.metal` file to be located in the same directory as the binary (i.e., `target/release/`). In future versions, this will likely be embedded in the binary itself.
- Evaluating a model with more than one token at a time is not currently supported in GGML's Metal implementation. An `llm` inference session will fall back to the CPU implementation (typically during the 'feed prompt' phase) but will automatically use the GPU once a single token is passed per evaluation (typically after prompt feeding).
- Not all model architectures will be equally stable when used with Metal due to ongoing work in the underlying implementation. Expect `llama` models to work fine though.
- With Metal, it is possible but not required to use `mmap`. As buffers do not need to be copied to VRAM on M1, `mmap` is the most efficient however.
- Debug messages may be logged by the underlying GGML Metal implementation. This will likely go away in the future for release builds of `llm`.

## Debugging

This repository includes a [`launch.json` file](../.vscode/launch.json) that can
be used for
[debugging with Visual Studio Code](https://code.visualstudio.com/docs/editor/debugging) -
this file will need to be updated to reflect where models are stored on your
system. Debugging with Visual Studio Code requires a
[language extension](https://code.visualstudio.com/docs/languages/rust#_install-debugging-support)
that depends on your operating system. Keep in mind that debugging text
generation is extremely slow, but debugging model loading is not.

## LLM References

Here are some tried-and-true references for learning more about large language
models:

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - an
  excellent technical description of how this seminal language model generates
  text
- [Andrej Karpathy's "Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) -
  a series of in-depth YouTube videos that guide the viewer through creating a
  neural network, a large language model, and a fully functioning chatbot, from
  scratch (in Python)
- [rustygrad](https://github.com/Mathemmagician/rustygrad) - a native Rust
  implementation of Andrej Karpathy's micrograd
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Chapter 12
  specifically)
