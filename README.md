# learnvulkanhpp

本仓库是一个以 **Vulkan-HPP**（官方 C++绑定）为核心的学习与实验项目，使用 **C++23 模块**（`import vulkan`）配合 **RAII** 封装（`vk::raii`）从零搭建图形程序，示例从创建实例、枚举设备与队列到命令缓冲等基础流程逐步展开。

## 技术要点

- **Vulkan**：通过 SDK 自带的 `vulkan.cppm` 作为 C++ 模块引入；启用动态调度（`VULKAN_HPP_DISPATCH_LOADER_DYNAMIC`），运行时加载函数指针，无需静态链接 Vulkan 库。
- **构建**：CMake 4.1+，编译器需支持C++23，可使用 
  - clang21+, `-stdlib=libc++`，
  - msvc
- **示例依赖**：通过 Conan 获取 GLFW（窗口与表面）、GLM（数学）、SPIRV-Reflect 和 TinyObjLoader（OBJ 载入），公共代码放在 `common` 模块中。

## 仓库结构（简要）


| 目录          | 说明                                    |
| ----------- | ------------------------------------- |
| `examples/` | 可执行示例（如 `basic`：初始化 Vulkan 上下文与设备等） |
| `common/`   | 共享 C++ 模块                             |
| `3rdparty/` | 第三方依赖的 CMake 适配层                       |

当前示例：

- `basic`：创建 Vulkan 实例、设备、队列和命令缓冲。
- `triangle`：通过光栅管线向交换链绘制三角形。
- `geometry`：使用 Geometry Shader 将输入点扩展为彩色三角形；需要 GPU 支持 `geometryShader` Feature。
- `tessellation`：使用 Tessellation Control/Evaluation Shader 将 quad patch 细分为动态起伏的线框曲面；需要 GPU 支持 `tessellationShader` 和 `fillModeNonSolid` Feature。
- `model`：使用 TinyObjLoader 载入 `assets/models/bunny.obj`，为没有法线的网格生成平滑法线，并通过顶点/索引缓冲进行光照渲染。左键拖动旋转，中键拖动平移，滚轮缩放，`R` 重置视角。
- `shadow`：使用方向光 2D Shadow Map 进行两遍光栅化，内置平面和立方体展示硬阴影，并始终在右上角显示深度图预览。左键拖动旋转，中键拖动平移，滚轮缩放，`R` 重置视角。
- `raytracing`：使用 KHR Ray Tracing Pipeline、BLAS/TLAS 和 SBT 绘制光追三角形；需要支持 Vulkan 1.2、buffer device address、acceleration structure 和 ray tracing pipeline 的 GPU。
- `rayquery`：使用 BLAS/TLAS 和 `VK_KHR_ray_query` 在 fragment shader 中查询静态场景的硬阴影；不使用 KHR Ray Tracing Pipeline 或 SBT，需要支持 Vulkan 1.2、buffer device address、acceleration structure 和 ray query 的 GPU。

## 构建

使用 Conan 2 安装依赖并生成 CMake preset：

```powershell
conan install . --output-folder=build -s build_type=Debug -s compiler.cppstd=23 -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
cmake --preset conan-debug
cmake --build --preset conan-debug
```

使用前请安装 **Vulkan SDK** 环境，并确保已安装验证层等可选组件以便调试。
