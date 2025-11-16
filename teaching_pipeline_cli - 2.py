# teaching_pipeline_cli.py
# -*- coding: utf-8 -*-
"""
多 Agent 教学流水线 Demo：
Teacher Agent  → PPT Agent → Checker Agent
并支持导出 TXT / Word / PPT / Excel
并加入 Meta-Agent 自我反思与下一步建议。

依赖：
- general_graph_agent_v1.py   （使用里面的 GraphMemory）
- export_utils.py
- openai 官方新版 SDK
"""

import os
from typing import Dict, Any, List

from openai import OpenAI

# 复用之前的记忆模块
from general_graph_agent_v1 import GraphMemory

# 复用导出工具
from export_utils import (
    export_to_txt,
    export_to_word,
    export_to_ppt,
    export_to_excel,
)

_client = OpenAI()


# =============== 通用 LLM 调用封装 ===============

def call_llm(system_prompt: str, user_content: str,
             model: str = "gpt-4.1-mini",
             temperature: float = 0.4) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content or ""


# =============== Teacher / PPT / Checker 三个 Agent ===============

def run_teacher_agent(task: str,
                      course_title: str,
                      student_level: str,
                      duration: str) -> str:
    """
    教学设计专家，用来生成详细教案。
    """
    system_prompt = """
你是一名经验丰富的教学设计专家，擅长为中职和大专院校教师设计结构化的课堂教学方案。

请你根据用户提供的课程信息，输出一份【完整的课堂教案】，要求：
1. 语言用中文，专业但通俗，教师可直接拿来上课；
2. 结构尽量清晰，建议使用如下模块（可根据需要微调）：
   【课程基本信息】
   【教学目标】（知识、技能、态度 三个维度）
   【教学重难点】
   【教学过程】
       - 导入（时间、步骤、师生活动）
       - 新授（分环节列出）
       - 学生活动（讨论 / 实操 / 演示等）
       - 小结
       - 作业或拓展任务
   【教学评价设计】
   【教学资源与准备】
3. 明确写出每一个环节的大致用时（分钟）。
4. 默认教学对象为职业院校学生，不要太理论化，要有具体活动和问题设计。
"""
    user_content = f"""
【课程主题】{course_title}
【学生对象】{student_level}
【课时长度】{duration}
【教师任务要求】{task}

请按上面的结构生成详细教案。
"""
    return call_llm(system_prompt, user_content)


def run_ppt_agent(teacher_plan: str,
                  course_title: str) -> str:
    """
    PPT 课件设计 Agent，只负责从教案生成 PPT 结构。
    """
    system_prompt = """
你是一名教学 PPT 设计专家，擅长把教案内容拆解成清晰的 PPT 结构。

现在给你一份详细教案，请你为这节课设计一套【教学 PPT 结构】，要求：
1. 用中文输出；
2. 只设计 PPT 的结构，不要写成完整讲稿；
3. 建议 8–12 页 PPT，每一页包含：
   - 页码（第 X 页）
   - 标题
   - 3–6 个项目符号要点（要点用短句、关键词，不要大段文字）
4. 明确标出哪些页面是“导入 / 过渡 / 总结 / 练习 / 提问”等功能页；
5. 尽量与教案中的教学环节一一对应。

输出格式示例（仅示例）：
第 1 页：
【标题】课程导入：走进人工智能
【要点】
- 提问：你日常生活中见过哪些“看起来很聪明”的机器？
- 视频或图片导入（1 分钟短片）
- 引出本节课主题和问题

第 2 页：
【标题】学习目标与课程安排
【要点】
- 本节课的 3 个学习目标
- 课程环节概览
- 学生需要完成的任务
...
"""
    user_content = f"""
下面是一份教案，请据此设计 PPT 结构。

【教案全文】：
{teacher_plan}
"""
    return call_llm(system_prompt, user_content)


def run_checker_agent(teacher_plan: str,
                      ppt_plan: str,
                      course_title: str,
                      student_level: str) -> str:
    """
    教研员 / 质检 Agent：审查教案 + PPT，并给出改进建议。
    """
    system_prompt = """
你是一名严谨的教研员兼教学督导，负责审核职业院校的教学设计质量。

现在有一套“教案 + PPT 结构”设计，请你从专业角度进行审查，并给出改进建议。
请用中文，结构化输出，包含至少以下部分：

【整体评价】
- 简要评价这套教案和 PPT 的整体质量（优点 + 总体建议）

【教学目标与重难点匹配度】
- 教学目标是否清晰、可操作？
- 重难点是否体现到教学过程和 PPT 要点中？

【教学过程与课堂活动】
- 各环节时间分配是否合理？
- 是否有足够的学生活动（讨论、实践、展示等）？
- 对中职 / 大专学生来说，难度是否合适？

【PPT 结构与呈现】
- PPT 页数与逻辑是否合理？
- 导入 / 讲解 / 练习 / 总结等功能页是否齐全？
- 是否有信息过载或过于零散的地方？

【可操作的改进建议】
- 用条目列出 5–10 条具体可操作的改进建议
- 建议中可以包含“某一环节增加一个问题 / 活动”“压缩某部分时间”“增加一个案例或演示”等
"""
    user_content = f"""
【课程主题】{course_title}
【学生对象】{student_level}

【教案】：
{teacher_plan}

【PPT 结构】：
{ppt_plan}
"""
    return call_llm(system_prompt, user_content)


# =============== 导出封装 ===============

def ask_export_pipeline(result: Dict[str, Any]):
    """
    教学流水线版本的导出逻辑。
    """
    task = result["task"]
    course_title = result["course_title"]
    student_level = result["student_level"]
    duration = result["duration"]
    teacher_plan = result["teacher_plan"]
    ppt_plan = result["ppt_plan"]
    checker_report = result["checker_report"]

    basename = f"teaching_pipeline_{os.getpid()}"

    print("\n是否导出本次【Teacher → PPT → Checker】结果？")
    print("a) 仅导出 TXT")
    print("b) 导出 TXT + Word")
    print("c) 导出 TXT + Word + PPT")
    print("d) 导出 TXT + Word + PPT + Excel")
    print("e) 不导出，返回主菜单")

    choice = input("请选择（a/b/c/d/e）：").strip().lower()
    if choice == "e" or not choice:
        print("不导出，返回主菜单。")
        return

    # 拼统一文本
    parts = [
        "【任务说明】",
        task,
        "\n【课程基本信息】",
        f"课程主题：{course_title}",
        f"学生对象：{student_level}",
        f"课时长度：{duration}",
        "\n【Teacher Agent 生成的教案】",
        teacher_plan,
        "\n【PPT Agent 生成的 PPT 结构】",
        ppt_plan,
        "\n【Checker Agent 审查与建议】",
        checker_report,
    ]
    all_text = "\n".join(parts)

    # 1. TXT 必导
    txt_path = export_to_txt(basename, all_text)

    # 2. Word
    if choice in ("b", "c", "d"):
        sections = {
            "任务说明": task,
            "课程基本信息": f"课程主题：{course_title}\n学生对象：{student_level}\n课时长度：{duration}",
            "Teacher 教案": teacher_plan,
            "PPT 结构": ppt_plan,
            "Checker 审查与建议": checker_report,
        }
        export_to_word(basename, "多 Agent 教学流水线结果", sections)

    # 3. PPT（简要汇总）
    if choice in ("c", "d"):
        slides = [
            {
                "title": "任务与课程概览",
                "bullets": [
                    f"课程：{course_title}",
                    f"学生：{student_level}",
                    f"时长：{duration}",
                ],
            },
            {
                "title": "Teacher 教案摘要",
                "bullets": [
                    "已生成完整教案（目标、重难点、过程、评价）。",
                    "详细内容见 Word / TXT。",
                ],
            },
            {
                "title": "PPT 结构摘要",
                "bullets": [
                    "已按教案设计 8–12 页教学 PPT 结构。",
                    "包含导入、讲解、活动、总结等功能页。",
                ],
            },
            {
                "title": "Checker 审查结论",
                "bullets": [
                    "从目标匹配度、活动设计、PPT 逻辑等维度给出评价。",
                    "提出 5–10 条可操作改进建议。",
                ],
            },
            {
                "title": "后续使用建议",
                "bullets": [
                    "教师可在此基础上进行个性化微调。",
                    "本流水线可多次迭代不同课次的教案与课件。",
                ],
            },
        ]
        export_to_ppt(basename, "多 Agent 教学流水线结果", slides)

    # 4. Excel：简单表格化整理
    if choice == "d":
        rows = [
            ["字段", "内容"],
            ["任务说明", task],
            ["课程主题", course_title],
            ["学生对象", student_level],
            ["课时长度", duration],
            ["Teacher 教案", teacher_plan],
            ["PPT 结构", ppt_plan],
            ["Checker 审查与建议", checker_report],
        ]
        export_to_excel(basename, rows)

    print("\n本次导出已完成。（TXT 路径：", txt_path, "）")


# =============== 流水线整体封装 ===============

def run_teaching_pipeline(task: str,
                          course_title: str,
                          student_level: str,
                          duration: str,
                          memory: GraphMemory) -> Dict[str, Any]:
    """
    完整执行：
    Teacher → PPT → Checker
    并把简略结果写入 memory（若可用）。
    """
    print("\n========== [Teacher Agent] 生成教案 ==========")
    teacher_plan = run_teacher_agent(task, course_title, student_level, duration)
    print(teacher_plan)

    print("\n========== [PPT Agent] 生成 PPT 结构 ==========")
    ppt_plan = run_ppt_agent(teacher_plan, course_title)
    print(ppt_plan)

    print("\n========== [Checker Agent] 审查与改进建议 ==========")
    checker_report = run_checker_agent(teacher_plan, ppt_plan, course_title, student_level)
    print(checker_report)

    # ------ 写入记忆：兼容不同 GraphMemory 实现，避免 AttributeError ------
    memory_item = {
        "task": task,
        "domain": "teaching_pipeline",
        "planner": teacher_plan[:4000],        # 防止太长
        "worker_output": ppt_plan[:4000],
        "critic_result": checker_report[:4000],
    }

    try:
        if hasattr(memory, "add"):
            memory.add(
                task=memory_item["task"],
                domain=memory_item["domain"],
                planner=memory_item["planner"],
                worker_output=memory_item["worker_output"],
                critic_result=memory_item["critic_result"],
            )
        elif hasattr(memory, "add_item"):
            memory.add_item(memory_item)
        elif hasattr(memory, "append"):
            memory.append(memory_item)
        elif hasattr(memory, "items") and hasattr(memory, "save"):
            try:
                memory.items.append(memory_item)
                memory.save()
            except Exception as inner_e:
                print(f"[Memory] 直接写入 items/save 失败：{inner_e}")
        else:
            print("[Memory] GraphMemory 中未发现可用的写入方法，已跳过记忆写入。")

    except Exception as e:
        print(f"[Memory] 写入记忆时出错，已忽略，不影响主流程：{e}")

    return {
        "task": task,
        "course_title": course_title,
        "student_level": student_level,
        "duration": duration,
        "teacher_plan": teacher_plan,
        "ppt_plan": ppt_plan,
        "checker_report": checker_report,
    }


# =============== Meta-Agent：自我反思与下一步建议 ===============

def run_pipeline_meta_agent(memory: GraphMemory, k: int = 5) -> str:
    """
    读取最近 k 条 teaching_pipeline 任务，做一次「自我反思 & 下一步规划」。
    """
    # 1. 尝试拿到原始 items
    items: List[Dict[str, Any]] = []
    try:
        raw_items = getattr(memory, "items", [])
        if isinstance(raw_items, list):
            # 只取 teaching_pipeline 域的
            items = [it for it in raw_items if isinstance(it, dict) and it.get("domain") == "teaching_pipeline"]
    except Exception as e:
        print(f"[Meta] 读取 memory.items 失败：{e}")

    recent_text = ""
    if items:
        recent = items[-k:]
        blocks = []
        for idx, it in enumerate(recent, 1):
            blocks.append(
                f"【第 {idx} 条任务】\n"
                f"任务：{it.get('task', '')}\n"
                f"教案摘要：{(it.get('planner') or '')[:400]}\n"
                f"PPT 摘要：{(it.get('worker_output') or '')[:400]}\n"
                f"审查摘要：{(it.get('critic_result') or '')[:400]}\n"
            )
        recent_text = "\n\n".join(blocks)
    else:
        # 兜底：尝试用 build_summary
        try:
            recent_text = memory.build_summary(k)
        except Exception as e:
            recent_text = "（当前暂无 teaching_pipeline 相关历史记录，或无法读取概要。）"
            print(f"[Meta] build_summary 调用失败：{e}")

    system_prompt = """
你是一个面向「教学 AI 应用研发」的 Meta-Agent，角色类似教研主任 + 产品负责人。

现在你要对最近若干次“教学流水线任务（Teacher → PPT → Checker）”做一次综合反思，并提出下一步改进建议。

请用中文、结构化输出，包含以下部分：

【一、最近教学流水线的整体画像】
- 从任务类型、课程主题、学生对象等角度，描述我们最近在做什么课、给谁上课
- 概括教案和 PPT 的共同特点（优点 + 潜在问题）

【二、从“教学质量”角度的建议】
- 给一线老师的 5–8 条具体建议（例如：目标表达、活动设计、评价方式、案例选择等）
- 建议尽量可操作，例如“把某一环节改成分组讨论”“增加一个项目作业模版”等

【三、从“AI 应用产品”角度的建议】
- 这条流水线离一个真正可用的产品还差什么？从功能、交互、稳定性、自动化程度等方面提出 5–8 条建议
- 尝试给出可以马上实施的“小步改进”（例如：增加一个参数、增加一个导出模版、增加一个快捷入口）

【四、下一步可执行的 3–5 个小任务】
- 面向开发者：列出 2–3 个可以在代码层面立刻尝试的改进任务（一句话 + 预期效果）
- 面向教学使用者：列出 1–2 个可以用现有流水线立刻试做的“实战场景”（例如：某门真实课程试用，或某类型课件批量生成）

要求：
- 不要空泛总结，要结合历史记录里的内容说话；
- 建议可以略带“路线图”味道，让后续开发有明确方向。
"""

    user_content = f"""
以下是最近若干次“教学流水线（Teacher → PPT → Checker）”的历史记录摘要（最多 {k} 条）：

{recent_text}

请根据上述信息，完成自我反思与下一步建议。
"""

    print("\n=============== [Meta-Agent 自我反思 & 下一步建议] ===============")
    meta_output = call_llm(system_prompt, user_content, model="gpt-4.1-mini", temperature=0.4)
    print(meta_output)
    return meta_output


# =============== CLI ===============

def print_menu():
    print("\n====== 多 Agent 教学流水线 Demo ======")
    print("1) 运行一次完整流水线（Teacher → PPT → Checker）")
    print("2) 查看最近任务概要（来自 GraphMemory）")
    print("3) 运行 Meta-Agent 自我反思（分析最近教学流水线）")
    print("4) 退出")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 警告：未检测到 OPENAI_API_KEY 环境变量。")
        print("请先设置环境变量后再运行。")

    memory = GraphMemory(path="teaching_pipeline_memory.json", max_items=10)

    while True:
        print_menu()
        choice = input("\n请选择操作编号：").strip()

        if choice == "1":
            print("\n请输入本次课程的信息（可大致填写）：")
            course_title = input("课程主题（例如：人工智能基础入门）：\n> ").strip()
            if not course_title:
                course_title = "未命名课程"

            student_level = input("学生对象（例如：中职一年级 / 大专一年级）：\n> ").strip()
            if not student_level:
                student_level = "中职一年级"

            duration = input("课时长度（例如：45 分钟 / 90 分钟）：\n> ").strip()
            if not duration:
                duration = "45 分钟"

            task = input("教师任务说明（例如：本节课是导入课 / 项目课 / 复习课 等）：\n> ").strip()
            if not task:
                task = f"为《{course_title}》设计一节 {duration} 的课堂教学方案。"

            result = run_teaching_pipeline(task, course_title, student_level, duration, memory)
            ask_export_pipeline(result)

        elif choice == "2":
            print("\n====== 最近任务概要 ======")
            try:
                print(memory.build_summary(10))
            except Exception as e:
                print(f"[Memory] 构建概要失败：{e}")

        elif choice == "3":
            try:
                run_pipeline_meta_agent(memory, k=5)
            except Exception as e:
                print(f"[Meta] 运行 Meta-Agent 时出错：{e}")

        elif choice == "4":
            print("已退出多 Agent 教学流水线 Demo。")
            break

        else:
            print("无效选择，请输入 1 / 2 / 3 / 4。")


if __name__ == "__main__":
    main()
