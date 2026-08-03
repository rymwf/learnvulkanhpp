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
- `marching_cubes`：使用两阶段 Compute Shader 先生成动态 Metaballs 3D 标量场，再通过经典 Marching Cubes 查表生成网格，并以 storage buffer 和 indirect draw 直接渲染。ImGui 面板可调整分辨率、iso level、动画速度、线框和 volume 切片，并显示 GPU 生成耗时与三角形数量；需要 graphics/present 队列同时支持 compute，以及 `R32_SFLOAT` storage image。相机操作与 `model` 相同。
- `shadow`：使用方向光 2D Shadow Map 进行两遍光栅化，内置平面和立方体展示硬阴影，并始终在右上角显示深度图预览。左键拖动旋转，中键拖动平移，滚轮缩放，`R` 重置视角。
- `shadow_filtering`：在同一场景中比较 Hard Shadow、PCF 和 PCSS。通过左上角 ImGui 面板切换模式并调整 Depth Bias、PCF Kernel 和 PCSS 搜索/半影参数，右上角始终显示原始深度图；相机控制与 `shadow` 相同。
- `shadow_csm`：使用三层深度纹理数组实现 Cascaded Shadow Maps；内置约 200 单位的户外地面与近、中、远多组遮挡物，使用 `20/70/200` 分割距离。主场景以轻微色调标出当前 cascade，右上角并排显示三个 cascade 的深度图。
- `shadow_vsm`：使用 Variance Shadow Maps 输出深度一阶/二阶矩，并执行全屏模糊 pass；右上角左侧为原始矩图，右侧为模糊后的矩图。
- `shadow_volume`：在基础 Shadow Map 上增加半透明体积区域，对光线方向上的采样点积分并按阴影衰减散射；右上角保留深度图预览。
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
