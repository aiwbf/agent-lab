# general_graph_cli_export.py
# -*- coding: utf-8 -*-
"""
通用 Graph Agent + 导出 Word / PPT / Excel 的 Demo

依赖：
- general_graph_agent_v1.py
- export_utils.py
"""

import os
from typing import Dict, Any

from general_graph_agent_v1 import GraphMemory, run_general_graph
from export_utils import (
    export_to_txt,
    export_to_word,
    export_to_ppt,
    export_to_excel,
)


def print_menu():
    print("\n====== 通用 Graph Agent 导出版 Demo ======")
    print("1) 教学模式（Teaching）")
    print("2) 通用模式（Generic）")
    print("3) 查看最近任务概要")
    print("4) 退出")


def ask_export(result: Dict[str, Any]):
    """
    在一次 Graph 任务完成后，询问是否导出。
    """
    task = result["task"]
    domain = result["domain"]
    plan = result["plan"]
    worker_output = result["worker_output"]
    critic_result = result["critic_result"]

    basename = f"{domain}_graph_{os.getpid()}"  # 简单一点，加上进程号防冲突

    print("\n是否导出本次结果？")
    print("a) 仅导出 TXT")
    print("b) 导出 TXT + Word")
    print("c) 导出 TXT + Word + PPT")
    print("d) 导出 TXT + Word + PPT + Excel")
    print("e) 不导出，返回主菜单")

    choice = input("请选择（a/b/c/d/e）：").strip().lower()

    if choice == "e" or not choice:
        print("不导出，返回主菜单。")
        return

    # 统一拼一个“总内容”，方便 TXT / Word 使用
    all_content = []
    all_content.append("【任务说明】")
    all_content.append(task)
    all_content.append("\n【领域】")
    all_content.append(domain)
    all_content.append("\n【Planner 结果】")
    all_content.append(plan)
    all_content.append("\n【Worker 输出】")
    all_content.append(worker_output)
    all_content.append("\n【Critic 审查】")
    all_content.append(critic_result)
    all_text = "\n".join(all_content)

    # 1. TXT 一定导
    txt_path = export_to_txt(basename, all_text)

    # 2. Word（可选）
    if choice in ("b", "c", "d"):
        sections = {
            "任务说明": task,
            "领域": domain,
            "Planner 结果": plan,
            "Worker 输出": worker_output,
            "Critic 审查": critic_result,
        }
        export_to_word(basename, "Graph Agent 任务结果", sections)

    # 3. PPT（可选，做一个非常简洁的结构）
    if choice in ("c", "d"):
        # 封装成 4 页内容页 + 1 页总结
        slides = [
            {
                "title": "任务说明 & 领域",
                "bullets": [f"任务：{task[:60]}...", f"领域：{domain}"],
            },
            {
                "title": "Planner 结果摘要",
                "bullets": [
                    "Planner 规划了任务步骤。",
                    "详细内容请见 Word / TXT。",
                ],
            },
            {
                "title": "Worker 输出摘要",
                "bullets": [
                    "Worker 生成了最终成果（教案 / 报告等）。",
                    "详细内容请见 Word / TXT。",
                ],
            },
            {
                "title": "Critic 审查结论",
                "bullets": [
                    "Critic 对结果做了质量评价。",
                    "详细意见请见 Word / TXT。",
                ],
            },
            {
                "title": "后续建议",
                "bullets": [
                    "可根据 Critic 建议进行人工微调。",
                    "亦可将此结果作为下一个 Graph 任务的输入。",
                ],
            },
        ]
        export_to_ppt(basename, "Graph Agent 任务结果", slides)

    # 4. Excel（可选）
    if choice == "d":
        rows = [
            ["字段", "内容"],
            ["任务说明", task],
            ["领域", domain],
            ["Planner 结果", plan],
            ["Worker 输出", worker_output],
            ["Critic 审查", critic_result],
        ]
        export_to_excel(basename, rows)

    print("\n本次导出已完成。（TXT 路径：", txt_path, "）")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("请先设置环境变量后再运行。")

    memory = GraphMemory(path="general_graph_memory.json", max_items=10)

    while True:
        print_menu()
        choice = input("\n请选择操作编号：").strip()

        if choice == "1":
            print("\n【教学模式】例如：为《人工智能基础入门》设计一节 45 分钟导入课。")
            task = input("请输入教学任务描述：\n> ").strip()
            if not task:
                print("未输入任务，返回主菜单。")
                continue
            result = run_general_graph(task, "teaching", memory)
            ask_export(result)

        elif choice == "2":
            print("\n【通用模式】例如：写一份关于本校 AI 教学改革的调研报告大纲。")
            task = input("请输入通用任务描述：\n> ").strip()
            if not task:
                print("未输入任务，返回主菜单。")
                continue
            result = run_general_graph(task, "generic", memory)
            ask_export(result)

        elif choice == "3":
            print("\n====== 最近任务概要 ======")
            print(memory.build_summary(10))

        elif choice == "4":
            print("已退出通用 Graph Agent 导出版 Demo。")
            break

        else:
            print("无效选择，请输入 1 / 2 / 3 / 4。")


if __name__ == "__main__":
    main()
