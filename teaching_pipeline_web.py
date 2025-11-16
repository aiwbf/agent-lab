import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from openai import OpenAI

from teaching_pipeline_cli import (
    run_teaching_pipeline,
    export_to_word,
    export_to_ppt,
    GraphMemory,
)

# ======================== 页面基础配置 & 样式 ========================

st.set_page_config(
    page_title="AI 教学流水线工作室",
    page_icon="🎓",
    layout="wide",
)

# 全局美化 CSS
st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top left, #f5f7ff 0, #ffffff 45%, #f7fbff 100%);
}
.app-header {
    padding: 1.4rem 1.6rem;
    border-radius: 1.2rem;
    background: linear-gradient(135deg, #2b90ff 0%, #5e60ff 40%, #9b5bff 100%);
    color: #ffffff;
    box-shadow: 0 14px 35px rgba(31, 84, 246, 0.26);
}
.badge {
    display: inline-block;
    padding: 0.12rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.3);
    margin-bottom: 0.35rem;
}
.card {
    background: #ffffff;
    border-radius: 1.1rem;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.section-title {
    font-weight: 600;
    font-size: 1.04rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.3rem;
}
.section-sub {
    font-size: 0.84rem;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding-top: 0.3rem;
    padding-bottom: 0.3rem;
}
.stDownloadButton button, .stButton button {
    border-radius: 999px !important;
    padding: 0.4rem 1.2rem;
}
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.timeline-item {
    padding: 0.55rem 0.9rem;
    border-radius: 0.8rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: #ffffff;
    margin-bottom: 0.6rem;
}
.timeline-meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.18rem;
}
.timeline-title {
    font-size: 0.96rem;
    font-weight: 600;
}
.timeline-summary {
    font-size: 0.84rem;
    color: #475569;
}
</style>
""",
    unsafe_allow_html=True,
)

# ======================== OpenAI 简单封装 ========================

DEFAULT_MODEL = "gpt-4.1-mini"
_client = OpenAI()


def call_llm(
    system_prompt: str,
    user_content: Union[str, Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    json_mode: bool = False,
    temperature: float = 0.4,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content
            if isinstance(user_content, str)
            else json.dumps(user_content, ensure_ascii=False),
        },
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ======================== GraphMemory 统一管理 ========================


def get_web_memory() -> GraphMemory:
    if "pipeline_memory" not in st.session_state:
        st.session_state["pipeline_memory"] = GraphMemory(max_items=20)
    return st.session_state["pipeline_memory"]


# ======================== 通用：渲染一个 section 的内容 ========================


def render_section_content(data: Any) -> None:
    """
    尽量把 run_teaching_pipeline 返回的内容“好看地展示”出来，
    兼容 str / list / dict 三种情况。
    """
    if data is None:
        st.info("（暂无数据）")
        return

    # 纯字符串：直接展示
    if isinstance(data, str):
        text = data.strip()
        if not text:
            st.info("（暂无数据）")
        else:
            st.write(text)
        return

    # 列表：可能是字符串列表，也可能是字典列表
    if isinstance(data, list):
        if not data:
            st.info("（暂无数据）")
            return

        # 全是字符串
        if all(isinstance(x, str) for x in data):
            for item in data:
                st.markdown(f"- {item}")
            return

        # 字典列表：尝试按 title / name + detail / content 展示
        if all(isinstance(x, dict) for x in data):
            for idx, item in enumerate(data, start=1):
                title = (
                    item.get("title")
                    or item.get("name")
                    or item.get("heading")
                    or f"条目 {idx}"
                )
                detail = (
                    item.get("detail")
                    or item.get("details")
                    or item.get("content")
                    or item.get("description")
                )
                st.markdown(f"**{idx}. {title}**")
                if detail:
                    if isinstance(detail, list):
                        for d in detail:
                            st.markdown(f"  - {d}")
                    else:
                        st.write(detail)
                st.write("")
            return

        # 其它混合情况：直接 json 展示
        st.json(data, expanded=False)
        return

    # 字典：按 key 分组输出
    if isinstance(data, dict):
        if not data:
            st.info("（暂无数据）")
            return
        for k, v in data.items():
            st.markdown(f"**▪ {k}**")
            if isinstance(v, (str, int, float)):
                st.write(v)
            else:
                render_section_content(v)
            st.write("")
        return

    # 其它类型：fallback
    st.json(data, expanded=False)


# ======================== 页面 1：教学流水线 ========================


def page_teaching_pipeline() -> None:
    memory = get_web_memory()

    # 顶部标题
    st.markdown(
        """
<div class="app-header">
  <div class="badge">AI Teaching Studio</div>
  <h1 style="margin-bottom:0.3rem;">🎓 一键生成教学设计流水线</h1>
  <p style="margin:0.1rem 0 0.3rem; font-size:0.98rem; opacity:0.94;">
    输入本次教学任务，AI 将帮助你自动完成 <b>教学目标 → 教学大纲 → 课堂活动 → 评价方式 → PPT 结构</b> 的全流程设计，
    并可一键导出 Word 教案与 PPT 结构稿。
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### 📝 填写本次教学任务")

        task = st.text_area(
            "用自然语言描述这次你想上的课（教学目标、重点难点、课堂风格等均可）：",
            height=160,
            placeholder="例如：为中职一年级学生设计一节 45 分钟的《人工智能基础入门》……",
        )

        c1, c2 = st.columns(2)
        with c1:
            course_title = st.text_input(
                "课程名称",
                value="人工智能基础入门",
                placeholder="例如：生成式 AI 应用初步 / Python 入门",
            )
        with c2:
            student_level = st.selectbox(
                "学生对象",
                [
                    "中职一年级",
                    "中职二年级",
                    "高一学生",
                    "高二学生",
                    "大一学生",
                    "大二学生",
                    "成人在职培训",
                ],
                index=0,
            )

        duration = st.selectbox(
            "课时长度",
            ["40 分钟", "45 分钟", "1 课时（约 45 分钟）", "90 分钟（两课时）"],
            index=2,
        )

        st.write("")
        generate = st.button("🚀 生成完整教学设计", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
<div class="section-title">🌈 使用建议</div>
<div class="section-sub">
- 尽量写清楚你想要的课堂效果，例如「希望有互动」「要有小组讨论」；  
- 课程名称和学生对象会直接进入教案和 PPT；  
- 生成完成后，可以在下方预览，也可以导出 <b>Word / PPT</b>。
</div>
<hr style="margin:0.6rem 0 0.5rem; opacity:0.4;" />
<div class="section-title">📌 最近一次任务概览</div>
""",
            unsafe_allow_html=True,
        )

        mem_list = memory.to_list()
        if not mem_list:
            st.info("当前还没有历史记录。先生成几次教案，历史画像会更有意义。")
        else:
            last = mem_list[-1]
            st.markdown(
                f"""
- 课程：**{last.get("course_title","")}**
- 对象：`{last.get("student_level","")}`
- 时长：`{last.get("duration","")}`
- 最近 PPT 页数：`{last.get("summary",{}).get("ppt_pages",0)}` 页  
- 时间：`{last.get("timestamp","")}`
"""
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 生成与展示结果
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    if generate:
        if not task.strip():
            st.warning("请先简单描述一下本次教学任务。")
            return

        with st.spinner("正在生成教学设计，请稍候…"):
            result = run_teaching_pipeline(
                task=task.strip(),
                course_title=course_title.strip() or "未命名课程",
                student_level=student_level,
                duration=duration,
                memory=memory,
            )

        st.session_state["last_result"] = result
        st.success("教学设计生成完成 ✅ 下方为本次 AI 生成的完整结构。")

    result = st.session_state.get("last_result")
    if not result:
        return

    # ====== 关键修正：智能适配字段名，避免“显示不正确” ======
    goals_data = (
        result.get("goals")
        or result.get("teaching_goals")
        or result.get("objectives")
        or result.get("targets")
    )

    outline_data = (
        result.get("content_outline")
        or result.get("outline")
        or result.get("teaching_outline")
        or result.get("syllabus")
        or result.get("content")
    )

    activities_data = (
        result.get("activities")
        or result.get("activity_design")
        or result.get("class_activities")
    )

    evaluation_data = (
        result.get("evaluation")
        or result.get("assessment")
        or result.get("evaluation_plan")
    )

    ppt_structure_data = (
        result.get("ppt_structure")
        or result.get("ppt_outline")
        or result.get("slides")
        or result.get("ppt")
    )

    st.write("")
    st.markdown("---")
    st.markdown("### 📚 本次 AI 教学设计结果")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🎯 教学目标",
            "📖 教学内容大纲",
            "🎲 课堂活动设计",
            "📊 评价方式",
            "🧩 PPT 结构预览",
            "🧾 原始结构(JSON)",
        ]
    )

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 教学目标</div>', unsafe_allow_html=True)
        render_section_content(goals_data)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📖 教学内容大纲</div>', unsafe_allow_html=True
        )
        render_section_content(outline_data)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🎲 课堂活动设计</div>', unsafe_allow_html=True
        )
        render_section_content(activities_data)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markmarkdown = st.markdown  # 避免 markmarkdown 拼写错误导致异常
        st.markdown(
            '<div class="section-title">📊 评价方式与作业</div>', unsafe_allow_html=True
        )
        render_section_content(evaluation_data)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🧩 PPT 结构预览</div>', unsafe_allow_html=True
        )
        if ppt_structure_data is None:
            st.info("当前结果中未找到 PPT 结构字段，导出 PPT 时会自动构造一个通用结构。")
        else:
            render_section_content(ppt_structure_data)
        st.markdown("</div>", unsafe_allow_html=True)

    # 原始 JSON 结构，方便你检查 run_teaching_pipeline 实际返回了什么
    with tab6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🧾 原始结果结构（调试用）</div>',
            unsafe_allow_html=True,
        )
        st.json(result, expanded=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 导出区保持不变 =====
    st.write("")
    st.markdown("### 💾 导出教学资源")

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown(
            '<div class="card"><div class="section-title">🧾 Word 教案（.docx）</div>',
            unsafe_allow_html=True,
        )
        try:
            word_bytes = export_to_word(result, to_file=False)
            if isinstance(word_bytes, bytes):
                st.download_button(
                    "📥 生成并下载 Word 教案",
                    data=word_bytes,
                    file_name="teaching_plan.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("当前运行在写文件模式，请稍后从命令行导出。")
        except Exception as e:
            st.error(f"导出 Word 失败：{e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_w2:
        st.markdown(
            '<div class="card"><div class="section-title">📽 PPT 结构稿（.pptx）</div>',
            unsafe_allow_html=True,
        )
        try:
            ppt_bytes = export_to_ppt(result, to_file=False)
            if isinstance(ppt_bytes, bytes):
                st.download_button(
                    "🎞 生成并下载 PPT 结构稿",
                    data=ppt_bytes,
                    file_name="teaching_ppt.pptx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "presentationml.presentation"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("当前运行在写文件模式，请稍后从命令行导出。")
        except Exception as e:
            st.error(f"导出 PPT 失败：{e}")
        st.markdown("</div>", unsafe_allow_html=True)


# ======================== 页面 2：教学目标抽取 ========================


def page_goal_extractor() -> None:
    st.markdown(
        """
<div class="app-header">
  <div class="badge">Goal Extractor</div>
  <h1 style="margin-bottom:0.3rem;">🎯 教学目标提取助手</h1>
  <p style="margin:0.1rem 0 0.3rem; font-size:0.98rem; opacity:0.94;">
    将已有的教案、课程描述或大纲粘贴进来，AI 帮你自动抽取、重写并结构化「知识目标 / 能力目标 / 情感态度价值观」三类教学目标。
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    raw_text = st.text_area(
        "请粘贴已有的教案、课程说明或教学内容片段：",
        height=260,
        placeholder="例如：本课程旨在让学生了解……",
    )

    col = st.columns([1, 1, 1])
    with col[0]:
        level = st.selectbox(
            "适用学段",
            ["小学", "初中", "高中", "中职", "高职 / 高校", "成人培训"],
            index=3,
        )
    with col[1]:
        subject = st.text_input("学科 / 模块", value="人工智能 / 信息技术")
    with col[2]:
        need_rewrite = st.checkbox("同时帮我优化用语", value=True)

    st.write("")
    do_extract = st.button("🎯 抽取并结构化教学目标", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not do_extract:
        return

    if not raw_text.strip():
        st.warning("请先粘贴一段教案或课程描述。")
        return

    with st.spinner("正在分析并抽取教学目标…"):
        sys_prompt = (
            "你是一名资深教学设计专家，请从用户提供的文本中抽取清晰的教学目标。"
            "输出 JSON，结构为："
            "{knowledge: [...], skills: [...], attitude: [...], rewriting: \"若需要优化的教学目标总述\"}。"
            "knowledge 侧重知识与概念，skills 侧重能力与方法，attitude 侧重情感态度价值观。"
            f"适用学段：{level}；学科/模块：{subject}。"
        )
        if not need_rewrite:
            sys_prompt += " 若用户不需要优化用语，则 rewriting 字段可以简要总结或留空。"

        resp = call_llm(sys_prompt, raw_text, json_mode=True)
        try:
            data = json.loads(resp)
        except Exception:
            data = {"raw": resp}

    st.write("")
    st.markdown("### 🎯 抽取结果")

    if isinstance(data, dict) and {"knowledge", "skills", "attitude"} <= data.keys():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📘 知识与理解目标**")
            for item in data.get("knowledge", []):
                st.markdown(f"- {item}")
        with c2:
            st.markdown("**🧩 技能与方法目标**")
            for item in data.get("skills", []):
                st.markdown(f"- {item}")
        with c3:
            st.markdown("**💡 情感态度与价值观**")
            for item in data.get("attitude", []):
                st.markdown(f"- {item}")

        if need_rewrite and data.get("rewriting"):
            st.markdown("---")
            st.markdown("#### ✍️ 推荐的「综合教学目标表述」")
            st.write(data.get("rewriting"))
    else:
        st.info("模型未按预期格式返回，下面显示原始结果：")
        st.json(data, expanded=False)


# ======================== 页面 3：历史任务 & Meta-Agent ========================


def page_meta_history() -> None:
    memory = get_web_memory()

    st.markdown(
        """
<div class="app-header">
  <div class="badge">Meta & History</div>
  <h1 style="margin-bottom:0.3rem;">🧠 历史教学任务分析 & 智能建议</h1>
  <p style="margin:0.1rem 0 0.3rem; font-size:0.98rem; opacity:0.94;">
    基于你最近通过「教学流水线」生成的任务记录，AI 将自动为你生成整体画像，并给出下一步优化教学设计和产品化的建议。
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("### 📜 最近任务时间线")

    items = memory.to_list()
    if not items:
        st.info("目前还没有任何历史记录。先去「一键生成教学设计流水线」页面生成几次任务吧。")
        return

    for item in reversed(items):
        st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="timeline-meta">🕒 {item.get("timestamp","")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="timeline-title">{item.get("course_title","未命名课程")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="timeline-summary">'
            f'对象：{item.get("student_level","")} · 时长：{item.get("duration","")} · '
            f'最近 PPT 页数：{item.get("summary",{}).get("ppt_pages",0)} 页'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Meta-Agent：教学画像 & 下一步建议")

    if "meta_result" not in st.session_state:
        st.session_state["meta_result"] = None

    if st.button("✨ 刷新智能分析与建议", type="primary"):
        with st.spinner("Meta-Agent 正在阅读你的历史记录并生成建议…"):
            sys_prompt = (
                "你是一名 AI 教学产品规划顾问。"
                "下面是一系列最近使用 AI 教学流水线生成的任务记录，请你："
                "1）先概括过去这些课程在主题、对象、课时上的整体画像；"
                "2）从提升教学质量的角度给出 5 条具体可执行建议；"
                "3）从“把它变成一个真正好用的 AI 教学产品”的角度，再给出 3–5 条下一步开发建议。"
                "要求分条，条目短、具体、可执行。"
            )

            user_content = {
                "history": items,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }

            meta_text = call_llm(sys_prompt, user_content, json_mode=False)
            st.session_state["meta_result"] = meta_text

    if st.session_state.get("meta_result"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### 📌 Meta-Agent 分析结果", unsafe_allow_html=True)
        st.write(st.session_state["meta_result"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("点击上方按钮，生成一份基于你历史记录的智能分析。")


# ======================== 主入口 ========================


def main() -> None:
    st.sidebar.markdown(
        """
<div class="sidebar-title">🎓 AI 教学流水线工作室</div>
<p style="font-size:0.86rem; color:#64748b;">
结合 <b>教案设计 + PPT 结构 + 目标抽取 + 历史分析</b> 的一站式工具。
</p>
""",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "功能导航",
        (
            "📚 一键生成教案流水线",
            "🎯 教学目标提取助手",
            "🧠 历史任务分析 & Meta-Agent",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "提示：先在「一键生成教案流水线」中多做几次任务，\n"
        "再来「历史任务分析 & Meta-Agent」看整体画像会更有意思。"
    )

    if page.startswith("📚"):
        page_teaching_pipeline()
    elif page.startswith("🎯"):
        page_goal_extractor()
    else:
        page_meta_history()


if __name__ == "__main__":
    main()
