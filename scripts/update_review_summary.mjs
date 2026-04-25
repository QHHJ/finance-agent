import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

function parseNdjsonLines(ndjsonText) {
  return ndjsonText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

const workbookPath = path.resolve(process.argv[2] ?? "workbook.xlsx");
const mode = (process.argv[3] ?? "inspect").toLowerCase();

const inputBlob = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(inputBlob);

if (mode === "inspect") {
  const sheetsInspect = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 8000,
  });

  console.log("=== SHEETS ===");
  console.log(sheetsInspect.ndjson);

  const sheetRows = parseNdjsonLines(sheetsInspect.ndjson);
  const sheetNames = sheetRows
    .filter((row) => row?.kind === "sheet" && typeof row?.name === "string")
    .map((row) => row.name);

  for (const sheetName of sheetNames) {
    const snapshot = await workbook.inspect({
      kind: "table,formula",
      sheetId: sheetName,
      range: "A1:Z120",
      include: "values,formulas",
      tableMaxRows: 120,
      tableMaxCols: 26,
      tableMaxCellChars: 120,
      maxChars: 20000,
    });
    console.log(`=== SHEET SNAPSHOT: ${sheetName} ===`);
    console.log(snapshot.ndjson);
  }
} else if (mode === "append") {
  const summarySheet = workbook.worksheets.getItem("Summary");
  const appendixRows = [
    ["补充阶段", "新增遇到的问题", "解决方式", "新增效果"],
    [
      "11. 分类关键词乱码",
      "规则词典中存在中文乱码/编码不一致，导致关键词路由命中率下降。",
      "统一文件为 UTF-8，修复乱码关键词并重新整理分类词典；冻结一版可追踪配置。",
      "关键词路由恢复可用，基础分类准确率回升，后续评测口径更稳定。",
    ],
    [
      "12. 首页说进流程但未执行",
      "对话回复显示“进入差旅流程”，但前端没有真正跳转，用户体感是“只会说不执行”。",
      "在对话结果中增加明确 enter_flow 动作信号，页面消费信号后立刻切换流程并 rerun。",
      "“进入差旅/材料”从文案承诺变成真实动作，减少误导回复。",
    ],
    [
      "13. 意图误判（问句被当修改）",
      "“哪些文件对应哪个槽位”类问句被误判成执行修改，触发错误回复。",
      "重排意图优先级：先识别问句，再判断轻量编辑；补充“哪些/对应/哪个/对不上”等问句特征词。",
      "问答命中率提升，用户问“对应关系”时能先解释而不是乱执行。",
    ],
    [
      "14. 对话模板味过重",
      "回复大量重复固定句式，缺少上下文引用，用户感知不像智能体。",
      "缩短固定话术，改为基于当前状态动态生成；仅保留必要安全提示，避免每轮重复引导语。",
      "对话更自然，信息密度提高，用户不再反复看到同一套模板。",
    ],
    [
      "15. 对话与上传入口割裂",
      "上传区和对话区分离，用户需要在多个区域来回操作，主界面显得杂乱。",
      "按 workbench 方案收敛入口：对话框支持文本+多文件，上传改为对话内动作。",
      "交互路径缩短，首页信息层级更清晰，学习成本下降。",
    ],
    [
      "16. 执行结果回写不透明",
      "用户下达修改后，只看到结论，看不到执行了什么，难以建立信任。",
      "执行 agent 输出结构化变更摘要，回传给对话 agent 用于可读反馈与右侧面板同步。",
      "每次修改都有“做了什么、改了哪条”的可解释结果，便于复核。",
    ],
    [
      "17. 多智能体接入后能力退化",
      "新项目初版多 agent 仅跑通框架，识别能力明显弱于旧版规则链路。",
      "保留旧版强识别能力并迁入 Travel Specialist Agent，减少中间 adapter 的功能损耗。",
      "在保留多 agent 架构的前提下恢复识别质量，避免“重构后变差”。",
    ],
    [
      "18. 端到端评测缺失",
      "优化过程主要靠主观对话体验，缺少可量化准确率和时延指标。",
      "建设差旅 test dataset，按分类准确率、槽位完整率、平均处理时长做统一评测。",
      "改动效果可被量化验证，后续迭代可基于指标做版本对比。",
    ],
  ];

  summarySheet.getRange("A20:D28").values = appendixRows;

  const overallCell = summarySheet.getRange("B15");
  const overallCurrent = overallCell.values?.[0]?.[0];
  if (typeof overallCurrent === "string" && !overallCurrent.includes("多智能体协同")) {
    overallCell.values = [[
      `${overallCurrent} 同时，已补齐多智能体协同落地中的关键问题修复（入口执行一致性、问答/执行分流、旧能力迁移），整体从“能演示”向“可验证、可追踪、可持续迭代”推进。`,
    ]];
  }

  const outputBlob = await SpreadsheetFile.exportXlsx(workbook);
  await outputBlob.save(workbookPath);
  console.log(`Updated workbook: ${workbookPath}`);
} else {
  console.error("Unsupported mode. Use inspect or append.");
  process.exit(1);
}
