# teaching_graph_agent_v4_multi.py
# -*- coding: utf-8 -*-
"""
第三阶段：多 Agent 协作版（Teacher → PPT → Checker）

功能：
- Teacher Agent：生成教学设计（课程名、目标、活动、资源、简介）。
- PPT Agent：根据教学设计生成 8 页 PPT 大纲。
- Checker Agent：对整体设计给出审查意见。
- 汇总为 teaching_result，并调用 teaching_exporter 导出 Word/PPT/Excel。
- 使用本地 Memory（teaching_memory.json）记住最近 10 次任务。

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

# 你可以根据自己的账号调整模型名称
DEFAULT_MODEL = "gpt-4.1-mini"  # 或 "gpt-5.1-mini" / "gpt-5.1"

client = OpenAI()

# =========================
# 一、简单 Memory 实现
# =========================

class SimpleMemory:
    """
    简单 Memory：
    - 将历史任务记录在本地 JSON 文件中
    - 每条记录结构：
      {
        "task": str,
        "plan": str,                # 这里可以放 Teacher + PPT 的综合说明
        "teaching_result": Dict,    # 最终用于导出的结构
        "critic_review": str        # Checker 的审查意见
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
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]
        self._save()
        print(f"[Memory] 已记录本次任务。当前共 {len(self.items)} 条记录。")

    def build_summary(self, max_tasks: int = 5) -> str:
        """
        为 LLM 构建一段简短历史摘要：
        - 做过哪些课程
        - 典型目标 + PPT 结构
        """
        if not self.items:
            return "（暂无历史记录，这是第一次执行任务。）"

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
# 二、统一 LLM 调用
# =========================

def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# =========================
# 三、Teacher Agent
# =========================

TEACHER_SYSTEM_PROMPT = """你是“Teacher Agent”，一名资深教学设计专家。

你的任务：
- 根据“当前教学任务”与“历史记忆”，设计本次课程的结构化教学方案。
- 产出必须是【合法 JSON】，结构如下（不要多字段、不要少字段）：

{
  "course_name": "课程名称（字符串）",
  "grade": "适用对象（如：中职一年级、大专一年级等）",
  "goals": ["教学目标1", "教学目标2", "教学目标3"],
  "key_points": ["知识要点1", "知识要点2", "知识要点3"],
  "activities": [
    "课堂活动1的完整文字说明（含目标、步骤、时间建议、材料）",
    "课堂活动2的完整文字说明",
    "课堂活动3的完整文字说明"
  ],
  "intro": "3-6 句话的课程简介，说明背景、目标、学习收获。",
  "resources": [
    "资源1说明（如推荐教材或网站）",
    "资源2说明",
    "资源3说明"
  ]
}

要求：
- JSON 顶层字段名必须是：course_name, grade, goals, key_points, activities, intro, resources。
- 严格输出 JSON，不要加任何“说明”“解释”“注释”。
- 所有字符串用双引号。
- 目标与活动适配任务中提到的教学对象（如中职/大专等）。
"""

def run_teacher_agent(task: str, memory: SimpleMemory) -> Dict[str, Any]:
    memory_summary = memory.build_summary()
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【历史记忆】\n{memory_summary}\n\n"
                f"【当前教学任务】\n{task}\n\n"
                "请按要求输出 JSON："
            ),
        },
    ]

    for attempt in range(3):
        content = call_llm(messages)
        print(f"\n========== [Teacher Agent 第 {attempt + 1} 次原始输出] ==========")
        print(content)
        try:
            result = json.loads(content)
            required_keys = [
                "course_name",
                "grade",
                "goals",
                "key_points",
                "activities",
                "intro",
                "resources",
            ]
            for k in required_keys:
                if k not in result:
                    raise ValueError(f"缺少字段：{k}")
            print("\n[Teacher Agent] JSON 解析成功。")
            return result
        except Exception as e:
            print(f"\n[Teacher Agent] JSON 解析失败：{e}")
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON 或字段不完整。请重新【只输出 JSON】，严格按照指定结构。",
                }
            )

    raise RuntimeError("Teacher Agent 连续 3 次未生成合法 JSON。")


# =========================
# 四、PPT Agent
# =========================

PPT_SYSTEM_PROMPT = """你是“PPT Agent”，专门负责把教学方案转成 8 页 PPT 结构。

你会收到：
- Teacher Agent 生成的教学方案 JSON（包括课程名、对象、目标、知识要点、活动、简介、资源）。

你的任务：
- 产出【合法 JSON】，结构如下：

{
  "ppt_outline": [
    {"title": "第1页标题", "bullets": ["要点1", "要点2", "要点3"]},
    {"title": "第2页标题", "bullets": ["要点1", "要点2", "要点3"]},
    {"title": "第3页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第4页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第5页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第6页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第7页标题", "bullets": ["要点1", "要点2"]},
    {"title": "第8页标题", "bullets": ["要点1", "要点2"]}
  ]
}

要求：
- 必须正好 8 个元素。
- 标题应覆盖：课程简介/背景、教学目标、核心概念、关键知识点、课堂活动、学习资源与延伸、总结与思考等。
- bullets 中写简洁的教学要点（每页 2-4 条）。
- 严格输出 JSON，不要任何多余文字。
"""

def run_ppt_agent(teacher_plan: Dict[str, Any]) -> Dict[str, Any]:
    pretty_plan = json.dumps(teacher_plan, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": PPT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "以下是 Teacher Agent 给出的教学方案 JSON：\n"
                f"{pretty_plan}\n\n"
                "请按要求输出 PPT 结构 JSON："
            ),
        },
    ]

    for attempt in range(3):
        content = call_llm(messages)
        print(f"\n========== [PPT Agent 第 {attempt + 1} 次原始输出] ==========")
        print(content)
        try:
            result = json.loads(content)
            if "ppt_outline" not in result:
                raise ValueError("缺少字段：ppt_outline")
            if not isinstance(result["ppt_outline"], list) or len(result["ppt_outline"]) != 8:
                raise ValueError("ppt_outline 必须是长度为 8 的列表")
            for slide in result["ppt_outline"]:
                if "title" not in slide or "bullets" not in slide:
                    raise ValueError("ppt_outline 中每个元素必须包含 title 与 bullets")
            print("\n[PPT Agent] JSON 解析成功。")
            return result
        except Exception as e:
            print(f"\n[PPT Agent] JSON 解析失败：{e}")
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON 或结构不符合要求。请重新【只输出 JSON】，严格按照指定结构。",
                }
            )

    raise RuntimeError("PPT Agent 连续 3 次未生成合法 JSON。")


# =========================
# 五、Checker Agent
# =========================

CHECKER_SYSTEM_PROMPT = """你是“Checker Agent”，负责对 Teacher Agent 与 PPT Agent 的产物进行专业审查。

你会看到：
- 当前教学任务
- Teacher Agent 生成的教学方案 JSON
- PPT Agent 生成的 PPT 结构 JSON
- 历史记忆摘要

请从以下维度简要评价（不超过 400 字）：
1. 教学目标是否明确、可操作，是否匹配对象（中职/大专等）。
2. 课堂活动是否与目标对齐，是否有足够操作细节。
3. PPT 结构是否覆盖关键知识点、活动、资源和总结，逻辑是否清晰。
4. 整体对初学者是否友好，有无过难或过空泛之处。
5. 给出 2-4 条具体、可执行的改进建议。

请用简体中文回答，可以用条目或短段落，不需要输出 JSON。
"""

def run_checker_agent(
    task: str,
    teacher_plan: Dict[str, Any],
    ppt_result: Dict[str, Any],
    memory: SimpleMemory,
) -> str:
    memory_summary = memory.build_summary()
    pretty_teacher = json.dumps(teacher_plan, ensure_ascii=False, indent=2)
    pretty_ppt = json.dumps(ppt_result, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【历史记忆】\n{memory_summary}\n\n"
                f"【当前教学任务】\n{task}\n\n"
                f"【Teacher Agent 教学方案 JSON】\n{pretty_teacher}\n\n"
                f"【PPT Agent 结构 JSON】\n{pretty_ppt}\n\n"
                "请按要求给出审查意见："
            ),
        },
    ]
    review = call_llm(messages)
    print("\n========== [Checker Agent 审查意见] ==========")
    print(review)
    return review


# =========================
# 六、汇总为 teaching_result + 导出
# =========================

def combine_to_teaching_result(
    teacher_plan: Dict[str, Any],
    ppt_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将多 Agent 结果合并为统一结构 teaching_result：
    {
      "course_name",
      "grade",
      "goals",
      "activities",
      "ppt_outline",
      "intro",
      "resources"
    }
    """
    teaching_result = {
        "course_name": teacher_plan.get("course_name", "未命名课程"),
        "grade": teacher_plan.get("grade", ""),
        "goals": teacher_plan.get("goals", []),
        "activities": teacher_plan.get("activities", []),
        "ppt_outline": ppt_result.get("ppt_outline", []),
        "intro": teacher_plan.get("intro", ""),
        "resources": teacher_plan.get("resources", []),
    }
    print("\n========== [汇总后的 teaching_result] ==========")
    print(json.dumps(teaching_result, ensure_ascii=False, indent=2))
    return teaching_result


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
# 七、主循环：多任务 + Memory
# =========================

def main():
    print("====== 教学 Graph Agent v4：多 Agent 协作（Teacher → PPT → Checker） ======")
    print("提示：输入“退出 / 结束 / q / quit”可离开。")
    memory = SimpleMemory(path="teaching_memory.json", max_items=10)

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("请先在系统中设置 OPENAI_API_KEY，再运行本脚本。")

    while True:
        task = input("\n请输入一个教学任务：\n> ").strip()
        if task.lower() in {"退出", "结束", "q", "quit"}:
            print("已退出。")
            break
        if not task:
            print("未输入任务，请重新输入。")
            continue

        # 1) Teacher Agent：教学设计
        teacher_plan = run_teacher_agent(task, memory)

        # 2) PPT Agent：结构化 8 页 PPT
        ppt_result = run_ppt_agent(teacher_plan)

        # 3) Checker Agent：质量审查
        checker_review = run_checker_agent(task, teacher_plan, ppt_result, memory)

        # 4) 汇总为 teaching_result
        teaching_result = combine_to_teaching_result(teacher_plan, ppt_result)

        # 5) 导出 Word / PPT / Excel
        export_all_files(teaching_result)

        # 6) 写入 Memory
        plan_summary = (
            "【Teacher Agent 教学方案】\n"
            + json.dumps(teacher_plan, ensure_ascii=False, indent=2)
            + "\n\n【PPT Agent 结构】\n"
            + json.dumps(ppt_result, ensure_ascii=False, indent=2)
        )
        memory.add_record(task, plan_summary, teaching_result, checker_review)

        print("\n本次多 Agent 协作任务已完成。你可以继续输入下一个任务，或输入“退出”结束。")


if __name__ == "__main__":
    main()
