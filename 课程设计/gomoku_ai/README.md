# 五子棋课程大作业

这是一个以“合法、稳定、可测试，再逐步提高棋力”为原则的 C 语言五子棋项目。当前已具备可完整进行一局的命令行双人对局。

## 目标

- 唯一硬性目标：交付能够完整下完一盘棋的 15×15 五子棋本地程序。
- 推荐起点：估值算法 + Minimax + Alpha-Beta。
- 正确处理黑方禁手、先后手、胜负和每步计时。
- 老师鼓励探索，可在投入允许时通过搜索优化或神经网络继续提高棋力。
- 在老师的 GitHub 课程仓库中持续展示开发、测试和调试过程。

## 文档入口

| 文档 | 用途 |
|---|---|
| [AGENTS.md](AGENTS.md) | 所有后续对话必须遵守的统一规范 |
| [项目背景](docs/PROJECT_CONTEXT.md) | 用户基础、目标和长期约束 |
| [需求与规则](docs/REQUIREMENTS.md) | 已确认要求、临时假设和待确认问题 |
| [方案比较](docs/SOLUTIONS.md) | 三套不同技术路线及推荐顺序 |
| [开发路线](docs/ROADMAP.md) | 从规则引擎到搜索增强的学习型里程碑 |
| [决策记录](docs/DECISIONS.md) | 已采纳、提议中及待决定的事项 |
| [项目进度](docs/PROGRESS.md) | 当前状态和下一次会话的起点 |
| [GitHub 工作流](docs/GIT_WORKFLOW.md) | 小步提交、推送和过程展示规则 |

## 当前状态

- 阶段：已完成 M1～M5 全流程与 **V3 单网增量 Pattern-NNUE（`GPN1`）+ 原生 `raylib` 桌面 GUI（`gomoku_gui`）**，目前正在 Windows RTX 4070 上进行 V3 第 11～22 轮加训。
- 已完成：15×15 棋盘与禁手规则、Alpha-Beta + VCF 搜索、V3 Pattern-NNUE & V2 双网纯 C11 推理、`raylib 5.5` 桌面图形界面、悔棋、可复盘棋谱保存以及双构建全量 CTest。
- 尚未完成：最终课堂提交接口打包确认。

课程仓库：<https://github.com/Zt66666cpu/C-Cpp-2026>

## 构建与运行

用 CLion 打开本项目目录并重新加载 `CMakeLists.txt`：
- 选择 **`gomoku_gui`** 目标运行 **原生桌面图形界面**（基于 `raylib 5.5` 自动静态构建，支持鼠标点击落子、黑方禁手红 `✕` 提示、V3/V2/启发式热切换、悔棋、AI 提示与实时胜率条；若无需 GUI 可加 `-DBUILD_GUI=OFF`）。
- 选择 **`gomoku_ai`** 目标运行命令行终端版本。

交互命令：
- 输入 `行 列` 落子，范围为 `1～15`，例如 `8 8`；
- 输入 `u` 或 `undo` 悔棋（人机模式回退两手，双人模式回退一手）；
- 输入 `q` 或 `quit` 退出（对局自动保存复盘记录到 `last_game.log`）。

常用命令行参数：
- `--ai-vs-ai --games N --depth D --time SEC`：自动对弈基准测试；
- `--nnue PATH --cnn PATH` / `--no-nn`：指定双网权重路径或退回纯手工启发式估值；
- `--save-record PATH`：导出可复盘棋谱与耗时日志到指定路径。

当前 macOS 命令行环境没有全局 `cmake`，CLion 可使用其内置 CMake。若终端已安装 CMake，也可执行：

```sh
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
./cmake-build-release/gomoku_gui   # 桌面图形界面
./cmake-build-release/gomoku_ai    # 命令行终端版本
```

若终端已安装 CTest，可运行自动测试：

```sh
ctest --test-dir cmake-build-debug --output-on-failure
```
