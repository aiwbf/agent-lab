# teaching_app_cli_meta.py
# -*- coding: utf-8 -*-
"""
教学 Graph Agent 应用版（Meta 增强版）：
- 自动教案 / 课件流水线
- + Meta-Agent：自我反思 & 自动规划下一步任务

依赖：
- teaching_graph_agent_v4_multi.py
- teaching_exporter.py
- openai>=1.0.0
- 已设置 OPENAI_API_KEY 环境变量
"""

import os
from typing import Dict, Any, List

from teaching_graph_agent_v4_multi import (
    SimpleMemory,
    run_teacher_agent,
    run_ppt_agent,
    run_checker_agent,
    combine_to_teaching_result,
    export_all_files,
)

# =========================
# 一、小型 LLM 调用封装（Meta-Agent 用）
# =========================

try:
    from openai import OpenAI
    _client = OpenAI()
except Exception as e:
    _client = None
    print("⚠️ 警告：未能初始化 OpenAI 客户端，请确认 openai 库已安装且环境变量 OPENAI_API_KEY 已设置。")
    print("错误信息：", e)


def call_meta_llm(system_prompt: str, user_content: str, model: str = "gpt-4.1-mini") -> str:
    """
    Meta-Agent 用的轻量 LLM 调用。
    - 默认模型改为 gpt-4.1-mini（你的环境已经在用）
    - 如果调用失败，返回一段提示文本，不让主程序崩溃
    """
    if _client is None:
        return "【Meta-Agent 提示】当前未正确配置 OpenAI 客户端，暂无法生成自动反思与学习建议。"

    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【Meta-Agent 提示】调用 LLM 失败：{e}"


# =========================
# 二、教学模板定义（同前一版，可继续扩展）
# =========================

TEMPLATES = [
    {
        "id": "1",
        "name": "中职一年级 · AI 导入课（45 分钟）",
        "description": "面向零基础的中职一年级学生，做一节 45 分钟的人工智能导入课，侧重兴趣激发与基本概念。",
        "task_template": (
            "现在请为中职一年级学生设计一节 45 分钟的导入课，课程名称为《{course_name}》。"
            "学生对象：中职一年级，基础薄弱但对新技术感兴趣。"
            "本节课目标：用通俗方式介绍人工智能的基本概念、典型应用场景，并通过 2-3 个课堂活动激发学习兴趣。"
            "请生成完整教学资源包：包括 3 条教学目标、3 个课堂活动（含目标、步骤、时间、材料）、课程简介、"
            "关键知识点和推荐资源，并控制难度适合中职一年级初学者。"
        ),
    },
    {
        "id": "2",
        "name": "大专一年级 · 《人工智能基础》单元课（90 分钟）",
        "description": "适合大专一年级的《人工智能基础》课程单元课，包含理论 + 简单实践。",
        "task_template": (
            "现在请为大专一年级学生设计一节 90 分钟的《{course_name}》单元课。"
            "学生对象：大专一年级，具备基本计算机操作能力，对 AI 有初步好奇。"
            "本节课需要：在 90 分钟内讲清楚 1-2 个核心概念（如机器学习、监督学习示例），"
            "并安排至少一个简单的小组实践或案例分析活动。"
            "请生成完整教学资源包：3 条教学目标、3 个课堂活动、课程简介、关键知识点和推荐资源，"
            "并注意实践环节时间分配。"
        ),
    },
    {
        "id": "3",
        "name": "教师培训 · AI 赋能教学工作坊（半天）",
        "description": "面向一线教师的 AI 赋能教学工作坊，强调示范与可操作性。",
        "task_template": (
            "现在请设计一场面向中小学/中职教师的《{course_name}》AI 赋能教学工作坊，时长约半天（3 小时）。"
            "对象：一线教师，信息化水平一般，希望学会用 AI 帮助备课、出题和设计活动。"
            "请生成完整教学资源包：3 条教学目标，3 个围绕“如何用 AI 备课、出卷、设计活动”的互动环节，"
            "课程简介、关键知识点与工具清单（可以包含 ChatGPT、PPT 生成工具等），强调“可立刻上手”的实用性。"
        ),
    },
]


# =========================
# 三、Meta-Agent：自我反思 & 下一步规划
# =========================

def _build_activities_brief(activities: Any, max_items: int = 3) -> List[Dict[str, Any]]:
    """容错处理 activities：既支持 list[dict] 也支持 list[str] / 其他。"""
    if not isinstance(activities, list):
        return []

    brief: List[Dict[str, Any]] = []
    for a in activities[:max_items]:
        if isinstance(a, dict):
            brief.append(
                {
                    "name": a.get("name") or a.get("title") or "",
                    "goal": a.get("goal") or a.get("objective") or "",
                    "time": a.get("time") or a.get("duration") or "",
                }
            )
        else:
            # 如果是 str 或其他类型，就简短记录一下
            text = str(a)
            brief.append(
                {
                    "name": text[:40],
                    "goal": "",
                    "time": "",
                }
            )
    return brief


def _build_ppt_titles(ppt_outline: Any, max_items: int = 8) -> List[str]:
    """容错处理 ppt_outline：支持 list[dict] 或 list[str]。"""
    if not isinstance(ppt_outline, list):
        return []

    titles: List[str] = []
    for s in ppt_outline[:max_items]:
        if isinstance(s, dict):
            title = s.get("title") or s.get("name") or ""
        else:
            title = str(s)
        titles.append(title)
    return titles


def run_meta_agent(task: str, teaching_result: Dict[str, Any], memory: SimpleMemory) -> str:
    """
    Meta-Agent 读取：
    - 当前 task 描述
    - 当前 teaching_result 简要结构
    - memory 中的最近任务概要

    输出：
    - 对本次设计的简短评价
    - 3 条“下一步建议”
    """
    history_summary = memory.build_summary(max_tasks=8)

    activities_brief = _build_activities_brief(teaching_result.get("activities"))
    ppt_titles = _build_ppt_titles(teaching_result.get("ppt_outline"))

    # 为了减少 token，把 teaching_result 做一个轻度摘要
    simple_result_view = {
        "course_title": teaching_result.get("course_title"),
        "level": teaching_result.get("level"),
        "duration": teaching_result.get("duration"),
        "objectives": teaching_result.get("objectives", []),
        "activities_brief": activities_brief,
        "ppt_outline_titles": ppt_titles,
    }

    system_prompt = (
        "你是一名懂教学设计、人工智能教育和成人学习理论的 Meta-Agent，"
        "负责对“教学 Graph Agent”产生的结果进行反思，并给出下一步学习 / 改进建议。\n\n"
        "要求：\n"
        "1. 先用 3-5 句话，对“本次教学设计”的优点和可能的不足做简要评价。\n"
        "2. 然后用条目形式给出 3 条“下一步建议”，每条都要具体、可执行，"
        "   可以是：改进本节课、扩展新的单元/活动、用户本人可以学习/实践的方向等。\n"
        "3. 语言用中文，风格专业但通俗，不要废话。\n"
        "4. 不要重复打印完整教案，只围绕“反思 + 下一步”展开。\n"
    )

    user_content = (
        "【本次任务描述】\n"
        f"{task}\n\n"
        "【本次教学结果简要结构】\n"
        f"{simple_result_view}\n\n"
        "【历史任务概要（最多 8 条）】\n"
        f"{history_summary}\n"
    )

    meta_output = call_meta_llm(system_prompt, user_content)
    return meta_output


# =========================
# 四、统一跑一遍流水线 + Meta-Agent
# =========================

def run_full_pipeline(task: str, memory: SimpleMemory) -> None:
    """
    给一个教学任务字符串：
    - Teacher Agent：教学设计
    - PPT Agent：8 页 PPT 结构
    - Checker Agent：审查与改进建议
    - 汇总 teaching_result → 导出 Word/PPT/Excel
    - 写入 Memory
    - Meta-Agent：基于历史 & 本次结果，给出反思与下一步建议
    """
    print("\n================= [开始执行教学流水线] =================")

    # 1) Teacher Agent
    teacher_plan = run_teacher_agent(task, memory)

    # 2) PPT Agent
    ppt_result = run_ppt_agent(teacher_plan)

    # 3) Checker Agent
    checker_review = run_checker_agent(task, teacher_plan, ppt_result, memory)

    # 4) 汇总结果
    teaching_result = combine_to_teaching_result(teacher_plan, ppt_result)

    # 5) 导出 Word / PPT / Excel
    export_all_files(teaching_result)

    # 6) 写入 Memory
    plan_summary = (
        "【Teacher Agent 教学方案】\n"
        + str(teacher_plan)
        + "\n\n【PPT Agent 结构】\n"
        + str(ppt_result)
    )
    memory.add_record(task, plan_summary, teaching_result, checker_review)

    print("\n================= [教学流水线执行完毕] =================")

    # 7) Meta-Agent：自我反思 + 下一步规划
    print("\n================= [Meta-Agent 自我反思 & 下一步建议] =================")
    meta_output = run_meta_agent(task, teaching_result, memory)
    print(meta_output)
    print("\n================= [本轮全部结束] =================")


# =========================
# 五、菜单功能
# =========================

def print_main_menu() -> None:
    print("\n====== 教学 Graph Agent 应用版（Meta 增强版）======")
    print("1) 使用教学模板快速生成教案 + 课件 + 资源表 + Meta 建议")
    print("2) 自由模式：自己输入教学任务 + Meta 建议")
    print("3) 查看历史任务概要（最近若干次）")
    print("4) 退出")


def choose_template() -> Dict[str, Any]:
    print("\n可用教学模板：")
    for t in TEMPLATES:
        print(f"{t['id']}) {t['name']} —— {t['description']}")
    while True:
        choice = input("\n请输入模板编号（或按 Enter 取消）：").strip()
        if choice == "":
            return {}
        for t in TEMPLATES:
            if t["id"] == choice:
                return t
        print("无效的编号，请重新输入。")


def handle_template_mode(memory: SimpleMemory) -> None:
    tmpl = choose_template()
    if not tmpl:
        print("已取消模板模式，返回主菜单。")
        return

    course_name = input("\n请输入课程/活动名称（例如：人工智能基础入门）：\n> ").strip()
    if not course_name:
        course_name = "未命名课程"

    task = tmpl["task_template"].format(course_name=course_name)
    print("\n========== [本次任务描述] ==========")
    print(task)

    confirm = input("\n是否开始执行该教学流水线？(y/n)：").strip().lower()
    if confirm not in {"y", "yes", "是"}:
        print("已取消执行，返回主菜单。")
        return

    run_full_pipeline(task, memory)


def handle_free_mode(memory: SimpleMemory) -> None:
    print("\n自由模式：你可以用自然语言描述任何教学任务。")
    print("示例：为《人工智能基础入门》设计一节 90 分钟的导入课，适合中职一年级学生。")
    task = input("\n请输入你的教学任务描述：\n> ").strip()
    if not task:
        print("未输入任务，返回主菜单。")
        return

    run_full_pipeline(task, memory)


def handle_history(memory: SimpleMemory) -> None:
    print("\n====== 历史任务概要 ======")
    summary = memory.build_summary(max_tasks=8)
    print(summary)


# =========================
# 六、主入口
# =========================

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("请先在系统中设置 OPENAI_API_KEY，再运行本脚本。")

    memory = SimpleMemory(path="teaching_memory.json", max_items=10)

    while True:
        print_main_menu()
        choice = input("\n请选择操作编号：").strip()

        if choice == "1":
            handle_template_mode(memory)
        elif choice == "2":
            handle_free_mode(memory)
        elif choice == "3":
            handle_history(memory)
        elif choice == "4":
            print("已退出教学 Graph Agent 应用（Meta 增强版），再见。")
            break
        else:
            print("无效选择，请输入 1 / 2 / 3 / 4。")


if __name__ == "__main__":
    main()
