# teaching_graph_agent_v2.py
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List

from openai import OpenAI
from teaching_exporter import (
    export_to_word,
    export_to_ppt,
    export_to_excel,
)

# 你可以根据自己账号的实际模型名称调整
DEFAULT_MODEL = "gpt-4.1-mini"  # 或 "gpt-5.1", "gpt-5.1-mini" 等

client = OpenAI()


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


# ========== 1. Planner：生成教学规划 ==========

PLANNER_SYSTEM_PROMPT = """你是一名资深教学设计专家，擅长为中职和大专课程设计结构化的教学规划。
用户会给出一个教学任务，请你用要点方式（不超过 10 条），从以下角度给出规划：
- 教学目标（知识/技能/态度）
- 核心内容模块
- 课堂活动类型
- PPT 结构
- 评价方式与资源准备

请用简体中文回答，使用有层次的条目。"""


def run_planner(task: str) -> str:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"教学任务：{task}"},
    ]
    plan = call_llm(messages)
    print("\n========== [Planner 规划结果] ==========")
    print(plan)
    return plan


# ========== 2. Worker：生成结构化 teaching_result ==========

WORKER_SYSTEM_PROMPT = """你是一名教学内容生成智能体，负责把教学任务和规划，整理成结构化的教学资源结果。

请务必输出【合法 JSON】，且 JSON 顶层结构固定如下（不要多字段，不要少字段）：

{
  "course_name": "课程名称（字符串）",
  "grade": "适用对象（如：中职一年级 / 大专）",
  "goals": ["教学目标1", "教学目标2", "教学目标3"],
  "activities": ["活动1的完整文字说明", "活动2的完整文字说明", "活动3的完整文字说明"],
  "ppt_outline": [
    {"title": "第1页标题", "bullets": ["要点1", "要点2", "要点3"]},
    {"title": "第2页标题", "bullets": ["要点1", "要点2"]},
    ...
  ],
  "intro": "一段简洁的课程简介，3-6 句。",
  "resources": ["资源1说明", "资源2说明", "资源3说明"]
}

要求：
- 严格输出 JSON，不要加任何注释、解释或多余文字。
- 字段名必须是上面给出的英文名。
- 所有字符串用双引号。
"""


def run_worker(task: str, plan: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": WORKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"教学任务：{task}\n\n以下是教学规划，请基于此生成 JSON：\n{plan}",
        },
    ]

    for attempt in range(3):
        content = call_llm(messages)
        print(f"\n========== [Worker 第 {attempt + 1} 次原始输出] ==========")
        print(content)

        # 尝试解析 JSON
        try:
            result = json.loads(content)
            # 做一点简单校验
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
            # 给 LLM 一个明确的纠错提示
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON 或字段不完整。请重新【只输出 JSON】，严格按照之前给定的结构。",
                }
            )

    raise RuntimeError("Worker 连续 3 次都未能生成合法 JSON。")


# ========== 3. Critic：审查教学结果质量 ==========

CRITIC_SYSTEM_PROMPT = """你是一名教学质量评价专家，负责审查教学设计是否合理。

请从以下几个维度简要评价（不超过 300 字）：
- 教学目标是否具体、可操作，是否匹配对象
- 课堂活动是否有助于达成目标，步骤是否清晰
- PPT 结构是否覆盖核心知识点、逻辑是否清楚
- 课程简介与资源是否适合“零基础 / 初级阶段”学生
- 可以给出 2-3 条改进建议

请用简体中文，以条目或短段落形式给出。"""


def run_critic(task: str, plan: str, result: Dict[str, Any]) -> str:
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"教学任务：{task}\n\n"
                f"教学规划：\n{plan}\n\n"
                f"教学结果 JSON：\n{pretty_json}"
            ),
        },
    ]
    review = call_llm(messages)
    print("\n========== [Critic 审查意见] ==========")
    print(review)
    return review


# ========== 4. 导出文件：Word / PPT / Excel ==========

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


# ========== 5. 主流程：简版 Graph Agent ==========

def main():
    print("====== 教学 Graph Agent 第二版：接入导出模块 ======")
    task = input("请输入一个教学任务（例如：为《人工智能基础入门》设计一份教学资源包）：\n> ").strip()
    if not task:
        print("未输入任务，已退出。")
        return

    # 1) Planner
    plan = run_planner(task)

    # 2) Worker（生成结构化结果）
    teaching_result = run_worker(task, plan)

    # 3) Critic（给出审查意见）
    run_critic(task, plan, teaching_result)

    # 4) 导出 Word / PPT / Excel
    export_all_files(teaching_result)

    print("\n全部流程结束。你可以在当前目录中查看生成的文件。")


if __name__ == "__main__":
    # 简单检查一下 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("你需要先在系统中设置 OPENAI_API_KEY，或修改代码使用其它方式传入。")
    main()
