# teaching_graph_agent_v3_memory.py
# -*- coding: utf-8 -*-
"""
第二版 Graph Agent（加强版）：
- 引入本地 Memory：记住最近若干次教学任务与结果
- 每次新任务会参考过去经验优化规划与生成
- 自动导出 Word / PPT / Excel

依赖：
    pip install openai python-docx python-pptx openpyxl
"""

import json
import os
from typing import Any, Dict, List

from openai import OpenAI
from teaching_exporter import (
    export_to_word,
    export_to_ppt,
    export_to_excel,
)

# 你可以根据账号实际情况修改模型
DEFAULT_MODEL = "gpt-4.1-mini"  # 或 "gpt-5.1-mini" / "gpt-5.1" 等

client = OpenAI()

# =========================
# 一、简单 Memory 实现
# =========================

class SimpleMemory:
    """
    一个极简 Memory：
    - 将历史任务记录在本地 JSON 文件中
    - 每条记录结构：
      {
        "task": str,
        "plan": str,
        "teaching_result": Dict,
        "critic_review": str
      }
    - 只保留最近 max_items 条
    """

    def __init__(self, path: str = "teaching_memory.json", max_items: int = 10):
        self.path = path
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Memory] 保存失败：{e}")

    def add_record(
        self,
        task: str,
        plan: str,
        teaching_result: Dict[str, Any],
        critic_review: str,
    ) -> None:
        record = {
            "task": task,
            "plan": plan,
            "teaching_result": teaching_result,
            "critic_review": critic_review,
        }
        self.items.append(record)
        # 只保留最近 max_items 条
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]
        self._save()
        print(f"[Memory] 已记录本次任务。当前共 {len(self.items)} 条记录。")

    def build_summary(self, max_tasks: int = 5) -> str:
        """
        为 LLM 构建一段简短的历史摘要，供 prompt 使用。
        重点交代：
        - 做过哪些课程
        - 常见教学目标/活动/PPT 模块
        """
        if not self.items:
            return "（暂无历史记录，这是第一次执行任务。）"

        # 只看最近 max_tasks 条
        recent = self.items[-max_tasks:]
        lines: List[str] = []
        lines.append(f"以下是最近 {len(recent)} 次教学设计历史概要：")
        for i, rec in enumerate(recent, 1):
            tr = rec.get("teaching_result", {})
            course_name = tr.get("course_name", "未命名课程")
            grade = tr.get("grade", "")
            goals = tr.get("goals", [])
            ppt_outline = tr.get("ppt_outline", [])

            goal_short = "；".join(goals[:2]) if goals else "（无）"
            ppt_titles = [s.get("title", "") for s in ppt_outline[:3]]
            ppt_short = " / ".join([t for t in ppt_titles if t])

            lines.append(
                f"- 历史任务{i}：课程《{course_name}》，对象：{grade}；"
                f"代表性目标：{goal_short}；PPT 结构示例：{ppt_short}"
            )

        return "\n".join(lines)


# =========================
# 二、LLM 调用封装
# =========================

def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """
    简单封装一次 LLM 调用，返回 content 文本
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# =========================
# 三、Planner：生成教学规划
# =========================

PLANNER_SYSTEM_PROMPT = """你是一名资深教学设计专家，擅长为中职和大专课程设计结构化的教学规划。

你会收到两部分信息：
1）历史记忆摘要：说明之前已经做过哪些课程设计；  
2）当前教学任务：说明这一次要设计的课程与要求。

请你：
- 在尊重当前任务要求的前提下，参考历史经验；
- 避免与历史课程完全重复，尽量有“递进”和“改进”；
- 用条目形式给出本次教学规划（不超过 12 条），建议结构：

1. 教学对象与起点分析
2. 本次课或本课程的总体教学目标（知识/技能/态度）
3. 核心内容模块（2-5 个）
4. 核心课堂活动（2-4 种形式）
5. PPT 结构与页数建议
6. 评价与反馈方式
7. 可借鉴的历史经验（简要指出从历史中吸收了什么）

请用简体中文回答。
"""


def run_planner(task: str, memory: SimpleMemory) -> str:
    memory_summary = memory.build_summary()
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【历史记忆】\n{memory_summary}\n\n"
                f"【当前教学任务】\n{task}"
            ),
        },
    ]
    plan = call_llm(messages)
    print("\n========== [Planner 规划结果] ==========")
    print(plan)
    return plan


# =========================
# 四、Worker：生成结构化 teaching_result
# =========================

WORKER_SYSTEM_PROMPT = """你是一名教学内容生成智能体，负责把教学任务和规划，整理成结构化的教学资源结果。

你会看到：
- 历史记忆摘要：说明之前课程是怎么设计的；
- 当前任务与规划：说明这次要做什么。

请在综合考虑的基础上，输出【合法 JSON】，顶层结构严格如下（不要多字段，不要少字段）：

{
  "course_name": "课程名称（字符串）",
  "grade": "适用对象（如：中职一年级 / 大专）",
  "goals": ["教学目标1", "教学目标2", "教学目标3"],
  "activities": ["活动1的完整文字说明", "活动2的完整文字说明", "活动3的完整文字说明"],
  "ppt_outline": [
    {"title": "第1页标题", "bullets": ["要点1", "要点2", "要点3"]},
    {"title": "第2页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第3页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第4页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第5页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第6页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第7页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第8页标题", "bullets": ["要点1", "要点2"]}
  ],
  "intro": "一段简洁的课程简介，3-6 句。",
  "resources": ["资源1说明", "资源2说明", "资源3说明"]
}

要求：
- 严格输出 JSON，不要加任何注释、解释或多余文字。
- 字段名必须是上面给出的英文名。
- 所有字符串用双引号。
- goals 建议 3 条左右即可，activities 3 条左右即可。
"""


def run_worker(task: str, plan: str, memory: SimpleMemory) -> Dict[str, Any]:
    memory_summary = memory.build_summary()
    messages = [
        {"role": "system", "content": WORKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【历史记忆】\n{memory_summary}\n\n"
                f"【当前任务】\n{task}\n\n"
                f"【本次教学规划】\n{plan}\n\n"
                "请按照要求输出 JSON："
            ),
        },
    ]

    for attempt in range(3):
        content = call_llm(messages)
        print(f"\n========== [Worker 第 {attempt + 1} 次原始输出] ==========")
        print(content)

        try:
            result = json.loads(content)
            required_keys = [
                "course_name",
                "grade",
                "goals",
                "activities",
                "ppt_outline",
                "intro",
                "resources",
            ]
            for k in required_keys:
                if k not in result:
                    raise ValueError(f"缺少字段：{k}")
            print("\n[Worker] JSON 解析成功。")
            return result
        except Exception as e:
            print(f"\n[Worker] JSON 解析失败：{e}")
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON 或字段不完整。请重新【只输出 JSON】，严格按照之前给定的结构。",
                }
            )

    raise RuntimeError("Worker 连续 3 次都未能生成合法 JSON。")


# =========================
# 五、Critic：审查教学结果质量
# =========================

CRITIC_SYSTEM_PROMPT = """你是一名教学质量评价专家，负责审查教学设计是否合理，并给出改进建议。

你会看到：
- 历史记忆摘要：说明之前的设计情况；
- 当前任务与规划；
- 当前生成的教学结果 JSON。

请从以下维度简要评价（不超过 400 字）：
1. 教学目标：是否具体、可操作，是否匹配对象（如：中职一年级）
2. 课堂活动：是否帮助达成目标，是否有足够的操作细节
3. PPT 结构：是否覆盖核心知识点、逻辑是否清晰，有无冗余或缺失
4. 课程简介与资源：是否适合零基础或初级阶段学生
5. 与历史设计相比：本次有哪些进步或需要保持的优点
6. 给出 2-4 条具体改进建议（可以是内容、活动设计或评价方式）

请用简体中文，以条目或短段落形式给出。
"""


def run_critic(task: str, plan: str, result: Dict[str, Any], memory: SimpleMemory) -> str:
    memory_summary = memory.build_summary()
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【历史记忆】\n{memory_summary}\n\n"
                f"【当前教学任务】\n{task}\n\n"
                f"【本次教学规划】\n{plan}\n\n"
                f"【本次教学结果 JSON】\n{pretty_json}"
            ),
        },
    ]
    review = call_llm(messages)
    print("\n========== [Critic 审查意见] ==========")
    print(review)
    return review


# =========================
# 六、导出文件
# =========================

def safe_filename(name: str) -> str:
    bad_chars = '\\/:*?"<>|'
    for c in bad_chars:
        name = name.replace(c, "_")
    return name.strip() or "未命名课程"


def export_all_files(teaching_result: Dict[str, Any]) -> None:
    base = safe_filename(teaching_result.get("course_name", "未命名课程"))
    word_name = f"{base}_教案.docx"
    ppt_name = f"{base}_课件.pptx"
    excel_name = f"{base}_资源清单.xlsx"

    export_to_word(teaching_result, word_name)
    export_to_ppt(teaching_result, ppt_name)
    export_to_excel(teaching_result, excel_name)

    print("\n========== [导出完成] ==========")
    print(f"- Word：{word_name}")
    print(f"- PPT ：{ppt_name}")
    print(f"- Excel：{excel_name}")


# =========================
# 七、主循环：支持多轮任务 + Memory
# =========================

def main():
    print("====== 教学 Graph Agent v3：加入 Memory（记住最近 10 次任务） ======")
    print("提示：输入“退出 / 结束 / q / quit”可离开。")
    memory = SimpleMemory(path="teaching_memory.json", max_items=10)

    # 简单检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("请先在系统中设置 OPENAI_API_KEY，再运行本脚本。")
        # 不直接 return，方便你如果在代码里面手动设 key 也能继续
    while True:
        task = input("\n请输入一个教学任务：\n> ").strip()
        if task.lower() in {"退出", "结束", "q", "quit"}:
            print("已退出。")
            break
        if not task:
            print("未输入任务，请重新输入。")
            continue

        # 1) Planner
        plan = run_planner(task, memory)

        # 2) Worker
        teaching_result = run_worker(task, plan, memory)

        # 3) Critic
        critic_review = run_critic(task, plan, teaching_result, memory)

        # 4) 导出文件
        export_all_files(teaching_result)

        # 5) 写入 Memory
        memory.add_record(task, plan, teaching_result, critic_review)

        print("\n本次任务流程已完成。你可以继续输入下一个任务，或输入“退出”结束。")


if __name__ == "__main__":
    main()
