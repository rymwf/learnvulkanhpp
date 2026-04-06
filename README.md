# learnvulkanhpp

本仓库是一个以 **Vulkan-HPP**（官方 C++绑定）为核心的学习与实验项目，使用 **C++23 模块**（`import vulkan_hpp`）配合 **RAII** 封装（`vk::raii`）从零搭建图形程序，示例从创建实例、枚举设备与队列到命令缓冲等基础流程逐步展开。

## 技术要点

- **Vulkan**：通过 SDK 自带的 `vulkan.cppm` 作为 C++ 模块引入；启用动态调度（`VULKAN_HPP_DISPATCH_LOADER_DYNAMIC`），运行时加载函数指针，无需静态链接 Vulkan 库。
- **构建**：CMake 4.1+，编译器需支持C++23，可使用 
  - clang21+, `-stdlib=libc++`，
  - msvc
- **示例依赖**：GLFW（窗口与表面）、GLM（数学），公共代码放在 `common` 模块中。

## 仓库结构（简要）


| 目录          | 说明                                    |
| ----------- | ------------------------------------- |
| `examples/` | 可执行示例（如 `1-basic`：初始化 Vulkan 上下文与设备等） |
| `common/`   | 共享 C++ 模块                             |
| `3rdparty/` | 第三方库与预编译依赖                            |


使用前请安装 **Vulkan SDK** 环境，并确保已安装验证层等可选组件以便调试。