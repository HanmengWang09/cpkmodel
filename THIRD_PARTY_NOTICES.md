# 来源与第三方说明

本仓库从用户现有的CardioPINN-Research-Agent项目中提取模型计算内核，并新增独立运行和导出入口。原项目数据、检验程序和Agent流程未迁入。

呼吸力学实现根据用户提供的 `Cardio-respiratory-model-main/source_files/lungMechanics.cc` 方程整理为容积状态形式。源作者为Bianca Maria Laudenzi、Caterina Dalmaso、Lucas Omar Muller（2025）。原源项目的Apache-2.0许可证保留于 `third_party/Apache-2.0.txt`；计算核原有来源注释与参数文件来源哈希均保留。所提取Python实现采用容积状态和解析导数形式，与原C++文件的实现形式不同。

主动/被动心脏模型参考CircAdapt 2409公开的Patch、Wall、Bag及TriSeg关系。相关来源网址保留在参数元数据中。本仓库不包含原生CircAdapt源码、二进制库或其独立对照程序。当前开发闭合包含显式有效时标和低张力平滑，不能视为原生CircAdapt完整替代。

NumPy、SciPy、Numba和Matplotlib为运行依赖，按各自许可证发布，未将这些库的源码或环境打包进本仓库。
