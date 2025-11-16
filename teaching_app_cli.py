# teaching_app_cli.py
# -*- coding: utf-8 -*-
"""
教学 Graph Agent 应用版：自动教案生成器 / 课件流水线（CLI 小应用）

依赖：
- teaching_graph_agent_v4_multi.py（上一版多 Agent 协作脚本）
- teaching_exporter.py
- 已安装 openai, python-docx, python-pptx, openpyxl 等

功能：
- 提供命令行菜单：
  1) 使用教学模板快速生成教案+课件+资源表
  2) 自由输入教学任务
  3) 查看历史任务概要（来自 teaching_memory.json）
  4) 退出

- 内部调用：
  - SimpleMemory
  - run_teacher_agent
  - run_ppt_agent
  - run_checker_agent
  - combine_to_teaching_result
  - export_all_files
"""

import os
from typing import Dict, Any

from teaching_graph_agent_v4_multi import (
    SimpleMemory,
    run_teacher_agent,
    run_ppt_agent,
    run_checker_agent,
    combine_to_teaching_result,
    export_all_files,
)


# =========================
# 一、教学模板定义
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
# 二、统一跑一遍流水线
# =========================

def run_full_pipeline(task: str, memory: SimpleMemory) -> None:
    """
    给一个教学任务字符串：
    - Teacher Agent：教学设计
    - PPT Agent：8 页 PPT 结构
    - Checker Agent：审查与改进建议
    - 汇总 teaching_result → 导出 Word/PPT/Excel
    - 写入 Memory
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

    print("\n================= [本次教学流水线执行完毕] =================")


# =========================
# 三、菜单功能
# =========================

def print_main_menu() -> None:
    print("\n====== 教学 Graph Agent 应用版：自动教案 / 课件流水线 ======")
    print("1) 使用教学模板快速生成教案 + 课件 + 资源表")
    print("2) 自由模式：自己输入教学任务")
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
    summary = memory.build_summary(max_tasks=5)
    print(summary)


# =========================
# 四、主入口
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
            print("已退出教学 Graph Agent 应用，再见。")
            break
        else:
            print("无效选择，请输入 1 / 2 / 3 / 4。")


if __name__ == "__main__":
    main()
