# Finance Agent Workbench

本项目是一个本地化财务报销 Agent 原型，当前重点覆盖差旅报销和材料费报销两个场景。项目目标不是做一个自由聊天机器人，而是让用户上传票据、支付凭证、订单截图等材料后，系统能够完成识别、分类、核对、追问、修正和导出。

当前版本已经从早期单文件 Streamlit 原型拆分为多模块结构，`streamlit_app.py` 只保留启动、路由和 Agent 回调注入，差旅处理、差旅页面、材料费页面、模型配置、聊天组件和运行指标都已拆到独立模块，便于后续维护。

## 核心能力

- 首页立案：用户可以直接描述报销目标，或拖入多个 PDF/图片材料，系统先做预检查和流程分流。
- 差旅报销：自动识别机票/交通票据、支付记录、机票明细、酒店发票、酒店支付记录、酒店订单截图等材料。
- 材料费报销：识别发票字段和明细行，支持质量提示、规格/项目名检查、对话式轻量修正。
- 多 Agent 协同：Conversation Agent 负责理解用户输入和生成自然回复，Execution Agent 负责执行低风险修改，Travel/Material Specialist Agent 负责领域任务。
- LLM + 规则兜底：OCR/抽取/分类优先使用本地模型，关键结果再通过结构化规则、字段守卫和人工修正保持可控。
- 任务工作台：左侧管理历史任务，中间对话和上传，右侧查看结果、缺件、金额核对和导出。

## 技术栈

- Python + Streamlit
- LangGraph 风格的 Agent 编排
- Ollama 本地模型调用
- Qwen2.5-VL 用于图片/PDF 视觉 OCR 与字段抽取
- Qwen2.5 Instruct 用于文本分类、对话理解和回复生成
- SQLite 存储任务数据
- FAISS / RAG 组件用于检索实验

## 当前推荐模型

默认按本地 Ollama 配置读取模型：

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_VL_MODEL=qwen2.5vl:3b
OLLAMA_TEXT_MODEL=qwen2.5:7b-instruct
OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
OLLAMA_TRAVEL_DOC_TEXT_MODEL=qwen2.5:7b-instruct
USE_OLLAMA_VL=1
```

建议先确认本地模型存在：

```powershell
ollama list
```

如缺少模型，可按需拉取：

```powershell
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:7b-instruct
```

## 项目结构

```text
finance-agent/
  streamlit_app.py                 # Streamlit 启动入口，只保留路由和依赖注入
  app/
    agents/                        # Agent 合约、编排器、对话/执行/领域 Agent
    services/
      travel_processing.py         # 差旅 OCR、分类、金额提取、槽位修正、差旅回复
      ollama_config.py             # Ollama 模型选择与运行状态面板
      parser.py                    # 文件解析入口
      extractor.py                 # 发票抽取服务
      exporter.py                  # 导出服务
      learning.py                  # 学习/记忆相关逻辑
    ui/
      home_router.py               # 首页立案和流程分流
      travel_workbench.py          # 差旅工作台 UI
      material_workbench.py        # 材料费工作台 UI
      chat_widgets.py              # 聊天框、文件拖拽、打字机效果、全局样式
      pending_actions.py           # 待确认动作队列
      agent_metrics.py             # Agent 运行指标
      task_hub.py                  # 任务列表和工作台状态
      workbench.py                 # 通用工作台组件
    usecases/                      # 首页、差旅、材料费等业务用例
    db/                            # SQLite 模型和仓储
    retrieval/                     # RAG 检索后端
  scripts/
    test_qwen25vl_invoice.py       # 命令行测试 Qwen2.5-VL 发票抽取
    travel_doc_classifier_benchmark.py
                                    # 差旅材料分类评测脚本
    travel_e2e_benchmark.py        # 差旅端到端评测脚本
    eval_travel_chat.py            # 差旅对话链路评测
    smoke_agents.py                # Agent 冒烟测试
    rebuild_faiss_index.py         # 重建 FAISS 索引
  test dataset/                    # 本地测试样例，按类别文件夹组织
  data/                            # 本地运行数据，不建议提交数据库/上传件/导出件
```

## 安装与运行

推荐使用 Conda 环境：

```powershell
conda create -n finance-agent python=3.11 -y
conda activate finance-agent
pip install -r requirements.txt
```

如果 PowerShell 中 `conda activate` 报 `_CE_M` / `_CE_CONDA` 相关错误，可先清理当前会话：

```powershell
Remove-Item Env:_CE_M -ErrorAction SilentlyContinue
Remove-Item Env:_CE_CONDA -ErrorAction SilentlyContinue
conda activate finance-agent
```

启动应用：

```powershell
streamlit run streamlit_app.py
```

## 使用方式

1. 打开首页立案页。
2. 在底部对话框描述报销目标，或直接拖入多份 PDF/图片。
3. 系统根据材料和目标判断进入差旅或材料费流程。
4. 在工作台中继续上传材料、追问缺件、查看金额核对结果。
5. 如果分类或金额不对，可以直接对话修正，例如：

```text
酒店支付记录.jpg 这个是酒店支付记录
c95a8...jpg 是去程机票明细
Snipaste_2026...png 金额改为 7792
现在还缺什么？
哪些金额对不上？
```

## 评测与调试脚本

### 测试 Qwen2.5-VL 发票抽取

```powershell
python scripts/test_qwen25vl_invoice.py
```

该脚本支持终端对话和传入文件，用于单独验证视觉模型对发票/订单/支付凭证的字段抽取能力。

### 差旅分类评测

```powershell
python scripts/travel_doc_classifier_benchmark.py --dataset "test dataset"
```

评测集按子文件夹作为真实标签，脚本不依赖文件名做分类判断。适合调试机票、酒店、支付记录、明细等容易混淆的材料。

### 差旅端到端评测

```powershell
python scripts/travel_e2e_benchmark.py
```

用于检查从材料识别、分类、分配到缺件/金额核对的整体效果。

### Agent 冒烟测试

```powershell
python scripts/smoke_agents.py
```

## 设计思路

### 为什么不是“用户输入 -> LLM -> 直接输出”

财务报销属于强约束场景，如果完全依赖大模型自由生成，容易出现：

- 分类不稳定
- 结果不可复现
- 输出格式不可控
- 修正动作发散
- 金额和缺件状态被编造

因此本项目采用更可控的方式：

- LLM 负责 OCR、字段抽取、文本分类、自然回复和复杂表达理解。
- 规则负责结果约束、字段守卫、金额校验、状态推进。
- Execution Agent 只执行结构化命令，不让 LLM 直接改状态。
- 对高风险操作进入待确认队列，用户确认后再执行。

### 多 Agent 协同

当前主要 Agent 角色：

- Conversation Agent：理解用户输入，生成自然回复，必要时产生命令。
- Execution Agent：执行低风险修改、批量修正、状态更新。
- Travel Specialist Agent：处理差旅材料识别、分配、缺件和金额核对。
- Material Specialist Agent：处理材料费发票明细、质量提示和修正建议。
- Orchestrator：统一调度各 Agent，避免 UI 直接耦合具体执行逻辑。

## 当前限制

- 当前仍是本地原型，适合演示和实验，不是生产级财务系统。
- OCR 和分类质量依赖本地模型、图片质量、PDF 可解析程度。
- 差旅去程/返程需要结合城市、日期、订单字段综合判断，单张材料很难绝对准确。
- 本地数据库和测试材料可能包含隐私或业务数据，提交 GitHub 前应谨慎筛选。

## Git 提交建议

建议提交源码、脚本和必要的示例说明，不建议提交以下内容：

- `.env`
- `node_modules/`
- `__pycache__/`
- `data/finance_agent.db`
- `data/uploads/`
- `data/exports/`
- 大量 benchmark 输出目录
- 含真实隐私信息的票据图片/PDF

## 后续规划

- 继续拆分 `travel_processing.py`，将 OCR、分类、槽位修正、金额核对拆成更小模块。
- 建立更稳定的差旅分类评测集。
- 增加分类置信度、混淆矩阵和平均处理时长统计。
- 优化首页和工作台 UI 的一致性。
- 完善导出和人工确认链路。
