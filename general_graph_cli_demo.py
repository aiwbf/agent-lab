# general_graph_cli_demo.py
# -*- coding: utf-8 -*-
"""
命令行 Demo：
- 让你在“教学模式”和“通用模式”之间切换
- 调用 general_graph_agent_v1.run_general_graph
"""

import os
from general_graph_agent_v1 import GraphMemory, run_general_graph


def print_menu():
    print("\n====== 通用 Graph Agent Demo ======")
    print("1) 教学模式（Teaching）")
    print("2) 通用模式（Generic）")
    print("3) 查看最近任务概要")
    print("4) 退出")


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
            run_general_graph(task, "teaching", memory)

        elif choice == "2":
            print("\n【通用模式】例如：写一份关于本校 AI 教学改革的调研报告大纲。")
            task = input("请输入通用任务描述：\n> ").strip()
            if not task:
                print("未输入任务，返回主菜单。")
                continue
            run_general_graph(task, "generic", memory)

        elif choice == "3":
            print("\n====== 最近任务概要 ======")
            print(memory.build_summary(10))

        elif choice == "4":
            print("已退出通用 Graph Agent Demo。")
            break

        else:
            print("无效选择，请输入 1 / 2 / 3 / 4。")


if __name__ == "__main__":
    main()
