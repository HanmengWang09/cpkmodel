# cpkmodel：心肺肾耦合模型

从 CardioPINN-Research-Agent 提取的独立集总参数模型，用于心衰机制研究的前向数值实验。一次联立求解 **53 个状态**，包含主动四腔心脏、五心壁力学、四个动态瓣膜、16 个血管池、双肾血流与滤过、体液和水钠反馈、呼吸力学及胸膜压耦合。

此目录包含模型、参数、普通 JSON 初态和运行入口。检验校验程序、参数反演、Agent 工作流、实验评分及历史结果不在本项目中。

## 运行

Windows PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'D:\data\OneDrive\Desktop\cpkmodel'
.\run_windows.ps1
```

默认执行 8.5 秒仿真，使用随附参考初态、Radau 求解器和 0.01 秒输出间隔。首次运行会编译 Numba 内核，之后复用缓存。无需 API Key。

运行一个 85 秒的心搏—呼吸共同周期，或选择 BDF：

```powershell
.\run_windows.ps1 -Duration 85
.\run_windows.ps1 -Duration 8.5 -Method BDF -NoPlot
```

使用构造启动初态、显式初态或自定义运行配置：

```powershell
.\run_windows.ps1 -Startup
.\run_windows.ps1 -InitialState '.\runs\某次运行\final_state.json'
.\run_windows.ps1 -Config '.\configs\default.json' -Duration 5
```

`-Out` 可指定尚不存在的输出目录；已有目录会被拒绝，避免覆盖结果。所有相对路径均相对于项目根目录。`-Startup` 是从构造种子起算，含启动瞬态。

脚本固定使用 `D:\anaconda\envs\biomodel\python.exe`，并在导入科学计算库前，将 `TEMP`、`TMP`、`MPLCONFIGDIR`、`NUMBA_CACHE_DIR`、Python 字节码及 pip 缓存指向本项目 `runs/runtime`。不创建虚拟环境。

依赖为 NumPy、SciPy、Numba 和 Matplotlib，兼容范围见 `requirements.txt`。整理时使用的现有环境版本为 Python 3.11、NumPy 2.3.5、SciPy 1.17.1、Numba 0.62.1、Matplotlib 3.11.2。环境已具备依赖时无需重装。确需安装时，在项目目录先设置 D 盘缓存：

```powershell
$taskRuntime = Join-Path $PWD 'runs\runtime'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
$env:TEMP = $taskRuntime
$env:TMP = $taskRuntime
$env:MPLCONFIGDIR = Join-Path $taskRuntime 'matplotlib'
$env:PIP_CACHE_DIR = Join-Path $taskRuntime 'pip-cache'
& 'D:\anaconda\envs\biomodel\python.exe' -m pip install -r requirements.txt
```

## 输出

每次运行写入 `runs/simulation_时间戳/`：

| 文件 | 内容 |
|---|---|
| `states.csv` | 53 个状态随时间的变化，名称与单位见 `configuration.json` |
| `pressures.csv` | 16 个血管池及 4 个心腔的压力，mmHg |
| `flows.csv` | 血管、瓣膜及肾脏通路流量，mL/s |
| `observables.csv` | GFR、肾血流、钠浓度、尿量、胸膜压等生理量 |
| `trajectory.npz` | 全部状态、压力、流量及生理量的 NumPy 数组，无 pickle |
| `configuration.json` | 本次物理参数、外界输入、初态、单位及求解设置 |
| `final_state.json` | 末端 53 状态，可作为后续仿真的初态 |
| `summary.json`、`运行说明.md` | 运行完成状态、时长、求解器统计与中文说明 |
| `waveforms.png` | 心室压力、容积、瓣流、GFR、肺泡容积与胸膜压，`-NoPlot` 时不生成 |

`runs/` 不上传 GitHub，运行后在本机生成。`summary.json` 的 `completed=true` 表示本次仿真及输出完成。

## 文件结构

```text
cpkmodel/
  heart_active/    主动心脏、动态瓣膜、53状态耦合及Numba计算核
  heart_passive/   心壁被动材料、心包与TriSeg几何基础
  circulation/    16血管池、压容关系和连接拓扑
  renal/          双肾血流与滤过
  fluid/          间质交换、淋巴、渗透与水钠反馈
  respiratory/    四气室、呼吸驱动、胸壁与胸膜压
  data/           53状态相位参考初态
  solver.py       物理状态的分段刚性前向积分
  runner.py       配置、命令入口及结果导出
configs/default.json
docs/模型说明.md
run_windows.ps1
requirements.txt
```

## Python 接口

在项目根目录，使用指定解释器并设置 D 盘缓存后：

```python
from dataclasses import replace
import numpy as np
from cpkmodel import (
    ActiveBoundary, ActiveCircuit, ActiveProtocol,
    load_reference_parameters, load_initial_state, simulate,
)

parameters = load_reference_parameters()
# 例如显式修改心室主动尺度：
# parameters = replace(parameters, ventricular_scale=0.7)
model = ActiveCircuit(parameters)
boundary = ActiveBoundary()
result = simulate(model, ActiveProtocol((), (boundary,)),
                  np.linspace(0., 5., 501), load_initial_state())
print(result.states.shape)  # (501, 53)
```

`ActiveProtocol((切换时间,), (初始输入, 切换后输入))` 支持分段输入；改变输入不会重置心搏或呼吸相位。参数是 dataclass，可用 `replace` 显式替换；血管、肾、体液、呼吸和被动心脏的固定参数随各模块 JSON 提供，主动心脏参数见 `heart_active/parameters.py`。

模型结构、53 状态及耦合关系见[模型说明](docs/模型说明.md)。

## 当前科学范围

当前为固定开发闭合下的集总参数 ODE 模型。肺部分包含呼吸力学和肺循环，尚无肺泡血气交换；也没有 PINN。主动心脏采用公开的有效激活时标及平滑低张力处理，尚未证明与原生 CircAdapt 完全等价。模型未完成健康成人联合校准和临床外部验证。

随附初态取自原项目 2026-10-25 固定模型参考工况的相位状态，保留累计心搏、呼吸相位。改变参数或外界输入后的轨迹不自动继承原工况的周期稳定性。参数文件中的历史路径和哈希仅用于注明来源，运行不会读取这些目录。

来源与第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
