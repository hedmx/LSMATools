# Contributing to LSMATOOLS

首先，感谢你愿意为 LSMATOOLS 项目做出贡献！欢迎任何形式的贡献，包括 bug 报告、功能建议、代码提交和文档改进。

## 📑 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [提交代码规范](#提交代码规范)
- [Pull Request 流程](#pull-request-流程)

---

## 行为准则

本项目采用 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则。请保持开放、尊重和包容的社区氛围。

---

## 如何贡献

### 1. 报告 Bug

如果你发现了 bug，请创建 Issue 并包含以下信息：

- **清晰的标题**：简明扼要地描述问题
- **复现步骤**：详细说明如何复现该问题
- **期望行为**：描述你期望发生什么
- **实际行为**：描述实际发生了什么
- **环境信息**：Python 版本、依赖库版本、操作系统等
- **截图/日志**：如果适用，附上相关截图或错误日志

### 2. 提出功能建议

我们欢迎新的功能想法！请创建 Issue 并标注为 `enhancement`，包含：

- **功能描述**：清晰描述你想要的功能
- **使用场景**：说明这个功能的使用场景
- **实现建议**：如果有具体的实现想法更好
- **替代方案**：是否考虑过其他解决方案

### 3. 提交代码

#### 准备工作

1. Fork 本仓库到你的 GitHub 账号
2. Clone 你的 fork 到本地：
   ```bash
   git clone https://github.com/YOUR_USERNAME/LSMATOOLS_git.git
   cd LSMATOOLS_git
   ```
3. 创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### 开发规范

**代码风格**：

- 遵循 [PEP 8](https://pep8.org/) Python 代码规范
- 使用 4 个空格缩进
- 行宽限制在 100 字符以内
- 函数和类添加类型提示（Type Hints）

**注释要求**：

```python
def find_endplates(scan_lines, img_raw):
    """
    Detect endplate candidates using state machine approach.
    
    Args:
        scan_lines: List of scan lines from cortical line 2
        img_raw: 2D water-only image array
    
    Returns:
        List of (row, col, ep_type) tuples for confirmed endplates
    
    Algorithm:
        1. Three-zone signal statistics (Otsu thresholding)
        2. State machine descent/ascent edge detection
        3. Arc-length clustering with 5mm window
    """
    # Implementation here
    pass
```

**测试要求**：

- 新增功能应包含相应的测试用例
- 确保现有测试通过
- 测试覆盖率不应降低

### 4. 改进文档

文档改进同样重要！包括：

- 修正拼写/语法错误
- 补充缺失的说明
- 添加使用示例
- 改进算法解释
- 翻译文档（中英互译）

---

## 开发环境设置

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### 3. 验证安装

```bash
python -c "import LSMATOOLS_CP; print('Installation OK')"
```

---

## 提交代码规范

### Git Commit Message

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 重构（不改变功能）
- `test`: 测试相关
- `chore`: 构建/工具配置

**示例**：

```bash
git commit -m "feat(anterior-edge): add dual-modal fusion for red line generation"
git commit -m "fix(endplate): correct arc-length clustering boundary condition"
git commit -m "docs(readme): update installation instructions"
```

### 代码质量检查

在提交前运行：

```bash
# 代码格式化
black LSMATOOLS_CP/

# 静态检查
flake8 LSMATOOLS_CP/

# 类型检查
mypy LSMATOOLS_CP/

# 单元测试
pytest tests/

# 测试覆盖率
pytest --cov=LSMATOOLS_CP tests/
```

---

## Pull Request 流程

### 1. 准备 PR

确保你的分支是最新的：

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/LSMATOOLS_git.git
git fetch upstream
git rebase upstream/main
```

### 2. 推送代码

```bash
git push origin feature/your-feature-name
```

### 3. 创建 PR

在 GitHub 上：

1. 点击 "New Pull Request" 按钮
2. 选择你的分支
3. 填写 PR 描述（使用下面的模板）
4. 请求 Review

### 4. PR 模板

```markdown
## Description
简要描述你的更改内容和目的

## Related Issue
关联的 Issue 编号（如：Fixes #123）

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## Screenshots (if applicable)
添加相关截图展示效果

## Additional Notes
其他需要说明的事项
```

### 5. Code Review

维护者会 review 你的代码并提出反馈：

- 积极回应 review 意见
- 及时修改并推送更新
- 保持友好的讨论氛围

### 6. 合并

PR 被接受后：

- 代码会被合并到主分支
- 你会被列为贡献者
- 感谢你的贡献！🎉

---

## 常见问题

### Q: 我可以只修改文档吗？
A: 当然可以！文档改进同样重要，欢迎任何文档相关的 PR。

### Q: 我的 PR 多久会被 review？
A: 通常在 1-2 周内，取决于项目的活跃度和 PR 的复杂度。

### Q: 如何成为维护者？
A: 持续贡献高质量的代码和 review，积极参与社区讨论。

---

## 致谢

感谢所有为 LSMATOOLS 做出贡献的开发者！你们的支持让这个项目变得更好。

---

最后更新：2026-03-09
