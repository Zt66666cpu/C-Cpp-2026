# 五子棋课程大作业

这是一个以“合法、稳定、可测试，再逐步提高棋力”为原则的 C 语言五子棋项目。当前已进入棋局内核的起步阶段。

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

- 阶段：M1 棋局内核与规则。
- 已完成：项目文档基线；纯 C11 与 CMake 配置；跨文件空棋盘显示骨架；macOS 下编译和运行验证。
- 尚未完成：坐标输入、合法落子、胜负与禁手判断、电脑 AI、最终提交接口确认。
- 下一步：把当前只显示空棋盘的骨架扩展为可重复落子并检查边界与重复位置的手动对局。

课程仓库：<https://github.com/Zt66666cpu/C-Cpp-2026>

## 构建

用 CLion 打开本项目目录并重新加载 `CMakeLists.txt`，选择 `gomoku_ai` 构建目标运行。

当前 macOS 命令行环境没有全局 `cmake`，CLion 可使用其内置 CMake。若终端已安装 CMake，也可执行：

```sh
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug
./cmake-build-debug/gomoku_ai
```
