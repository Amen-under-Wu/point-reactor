## 说明
本项目为点堆动力学方程数值求解程序。

## 构建方式
1. 确保自己安装了`rustup`环境。
2. 确保自己安装了`wasm-pack`工具（可通过`cargo install wasm-pack`安装）。
3. 在项目根目录下，运行`wasm-pack build --target web`，这将在`pkg/`目录下生成所需的wasm文件。
4. 运行`http-server`，之后访问显示的链接（一般为`http://127.0.0.1:8080/`）。

一个可访问的demo部署在[https://amen-under-wu.github.io/cx/nuke.html](https://amen-under-wu.github.io/cx/nuke.html)。
