# general_graph_agent_v1.py
# -*- coding: utf-8 -*-
"""
第二版 Graph Agent 核心（通用版）：
- 支持两种“领域 profile”：
  1）teaching：教学设计 / 课堂活动 / PPT 结构
  2）generic：通用任务规划 / 文案 / 报告 / 方案
- 结构：Planner -> Worker -> Critic (+ 简单 Memory)

不依赖你之前的 teaching_* 文件，可以单独使用。
"""

import os
from typing import List, Dict, Any, Optional

# -------- OpenAI 客户端封装 --------

try:
    from openai import OpenAI
    _client = OpenAI()
except Exception as e:
    _client = None
    print("⚠️ [general_graph_agent_v1] 未能初始化 OpenAI 客户端，请确认 openai 库已安装且环境变量 OPENAI_API_KEY 已设置。")
    print("错误信息：", e)


def call_llm(system_prompt: str, user_content: str, model: str = "gpt-4.1-mini", temperature: float = 0.4) -> str:
    """
    通用 LLM 调用封装。
    如果调用失败，返回错误提示字符串，而不是抛异常。
    """
    if _client is None:
        return "【LLM 调用失败】未正确初始化 OpenAI 客户端。"

    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【LLM 调用失败】{e}"


# -------- 简单 Memory（可复用在任意领域）--------


class GraphMemory:
    """
    通用 Memory：
    - 只记“最近 N 次任务”的简要内容
    - 不绑死教学领域，以后可以被任何 Agent 使用
    """

    def __init__(self, path: str = "general_graph_memory.json", max_items: int = 10):
        self.path = path
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self.items = []
            return
        try:
            import json
            with open(self.path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
        except Exception:
            self.items = []

    def _save(self):
        try:
            import json
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.items[-self.max_items :], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("⚠️ [GraphMemory] 保存失败：", e)

    def add_record(self, task: str, domain: str, planner_plan: str, worker_output: str, critic_result: str):
        self.items.append(
            {
                "task": task,
                "domain": domain,
                "planner_plan": planner_plan,
                "worker_output": worker_output,
                "critic_result": critic_result,
            }
        )
        self.items = self.items[-self.max_items :]
        self._save()

    def build_summary(self, max_tasks: int = 8) -> str:
        if not self.items:
            return "（暂无历史任务记录。）"
        lines = []
        for i, item in enumerate(self.items[-max_tasks:], start=1):
            lines.append(
                f"{i}. [领域: {item.get('domain', 'unknown')}] 任务摘要：{item.get('task', '')[:60]}"
            )
        return "\n".join(lines)


# -------- Planner / Worker / Critic：通用 + 领域 profile --------

def build_planner_prompt(domain: str) -> str:
    """
    不同领域的 Planner system prompt。
    目前支持：
    - teaching
    - generic
    """
    if domain == "teaching":
        return (
            "你是一个教学设计 Planner Agent，负责把用户的教学任务拆解成清晰的步骤。\n"
            "要求：\n"
            "1. 先用 1-2 句总结理解到的任务。\n"
            "2. 然后输出若干“step_xxx”行，每行一个具体步骤，尽量可执行。\n"
            "3. 尽量围绕：教学目标、教学内容、课堂活动、评价方式、PPT 结构、教学资源等维度来规划。\n"
            "4. 用中文输出。"
        )
    else:
        # 通用任务 Planner
        return (
            "你是一个通用任务规划 Planner Agent，负责把用户的任务拆解成清晰的工作步骤。\n"
            "要求：\n"
            "1. 先用 1-2 句总结你的理解。\n"
            "2. 然后输出若干“step_xxx”行，每行一个步骤，尽量具体、可执行。\n"
            "3. 根据任务类型自由规划：如调研、写作、分析、方案设计、表格整理等。\n"
            "4. 用中文输出。"
        )


def build_worker_prompt(domain: str) -> str:
    if domain == "teaching":
        return (
            "你是一个教学执行 Worker Agent，根据 Planner 给出的步骤，真正写出教学方案。\n"
            "要求：\n"
            "1. 不要再重复列“step”，而是直接给出最终教学成果。\n"
            "2. 对于教学类任务，一般需要：\n"
            "   - 3 条左右“教学目标”；\n"
            "   - 2-3 个课堂活动（写清目标、步骤、时间、材料）；\n"
            "   - 一段课程简介；\n"
            "   - 必要时给出 PPT 结构或知识点列表。\n"
            "3. 语言专业但通俗，适合教师直接拿去用。"
        )
    else:
        return (
            "你是一个通用执行 Worker Agent，根据 Planner 的步骤，生成最终可用的结果。\n"
            "要求：\n"
            "1. 不要再重复列“step”，而是直接产出任务所需内容，如：大纲、报告、方案、表格说明等。\n"
            "2. 尽量结构化表达（分条、分段），保证别人可以直接拿去用。\n"
            "3. 语言用中文，专业但通俗。"
        )


def build_critic_prompt(domain: str) -> str:
    return (
        "你是一个 Critic Agent，负责审查 Worker Agent 的输出质量。\n"
        "请你：\n"
        "1. 用 3-5 句话简要评价该答案是否满足用户任务。\n"
        "2. 指出 2-3 个可以改进的点（如：不够具体、缺少结构、缺少案例等）。\n"
        "3. 最后用一行给出总体结论：'结论：可接受' 或 '结论：需要重做'。\n"
        "仅用中文回答，不要复写原答案。"
    )


def run_planner(task: str, domain: str, memory: Optional[GraphMemory] = None) -> str:
    """
    Planner：基于 domain 做任务拆解。
    """
    history = memory.build_summary(5) if memory is not None else "（无历史）"
    system_prompt = build_planner_prompt(domain)
    user_content = (
        f"【当前任务】\n{task}\n\n"
        f"【最近任务概要】\n{history}\n\n"
        "请根据当前任务，输出详细的规划步骤。"
    )
    print("\n[Graph] 第 1 步：Planner 规划中...")
    plan = call_llm(system_prompt, user_content)
    print("\n[Planner 输出] ==========================")
    print(plan)
    print("========================================")
    return plan


def run_worker(task: str, domain: str, plan: str, memory: Optional[GraphMemory] = None) -> str:
    """
    Worker：根据 Planner 结果 + 任务描述，生成最终成果。
    这里只用一个 LLM 调用，不做复杂工具循环，方便你理解。
    """
    history = memory.build_summary(5) if memory is not None else "（无历史）"
    system_prompt = build_worker_prompt(domain)
    user_content = (
        f"【当前任务】\n{task}\n\n"
        f"【Planner 规划结果】\n{plan}\n\n"
        f"【最近任务概要】\n{history}\n\n"
        "请根据上述内容，直接产出可以使用的最终成果。"
    )
    print("\n[Graph] 第 2 步：Worker 执行中...")
    output = call_llm(system_prompt, user_content, temperature=0.5)
    print("\n[Worker 输出] ===========================")
    print(output)
    print("========================================")
    return output


def run_critic(task: str, domain: str, plan: str, worker_output: str, memory: Optional[GraphMemory] = None) -> str:
    """
    Critic：评价 Worker 输出，不再强制重试，只给人类参考。
    """
    history = memory.build_summary(5) if memory is not None else "（无历史）"
    system_prompt = build_critic_prompt(domain)
    user_content = (
        f"【当前任务】\n{task}\n\n"
        f"【Planner 规划结果】\n{plan}\n\n"
        f"【Worker 输出】\n{worker_output}\n\n"
        f"【最近任务概要】\n{history}\n\n"
        "请按照要求进行评价。"
    )
    print("\n[Graph] 第 3 步：Critic 审查中...")
    critic = call_llm(system_prompt, user_content)
    print("\n[Critic 输出] ==========================")
    print(critic)
    print("========================================")
    return critic


def run_general_graph(task: str, domain: str, memory: Optional[GraphMemory] = None) -> Dict[str, Any]:
    """
    统一入口：
    - domain: "teaching" 或 "generic"
    - 返回一个 dict，包含 plan / worker_output / critic_result
    """
    if domain not in {"teaching", "generic"}:
        domain = "generic"

    plan = run_planner(task, domain, memory)
    worker_output = run_worker(task, domain, plan, memory)
    critic_result = run_critic(task, domain, plan, worker_output, memory)

    if memory is not None:
        memory.add_record(task, domain, plan, worker_output, critic_result)

    return {
        "task": task,
        "domain": domain,
        "plan": plan,
        "worker_output": worker_output,
        "critic_result": critic_result,
    }
