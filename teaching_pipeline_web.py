# teaching_pipeline_web.py
"""
基于 Streamlit 的教学流水线 Web 界面

依赖 teaching_pipeline_cli.py 中的核心逻辑：
- run_teaching_pipeline
- export_to_word
- export_to_ppt
- export_to_excel
- GraphMemory
"""

import os
import tempfile
from datetime import datetime

import streamlit as st

from teaching_pipeline_cli import (
    run_teaching_pipeline,
    export_to_word,
    export_to_ppt,
    export_to_excel,
    GraphMemory,
)

# =========================
#  全局样式
# =========================


def set_page_style():
    st.set_page_config(
        page_title="AI 教学流水线助手",
        page_icon="📚",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        /* 全局背景和字体 */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .main {
            background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 60%);
        }
        /* 顶部标题区 */
        .top-title {
            padding: 0.8rem 0 0.5rem 0;
        }
        .top-title h1 {
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .subtitle {
            color: #666;
            font-size: 0.9rem;
        }
        /* 卡片样式 */
        .card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .card-header {
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 0.5rem;
        }
        .section-title {
            font-weight: 600;
            font-size: 1rem;
            margin-top: 0.6rem;
            margin-bottom: 0.25rem;
        }
        .key-label {
            font-weight: 600;
            color: #0f172a;
        }
        .small-hint {
            font-size: 0.8rem;
            color: #6b7280;
        }
        .result-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.06);
            color: #2563eb;
            font-size: 0.75rem;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
#  内存管理：GraphMemory
# =========================


def get_web_memory() -> GraphMemory:
    """
    使用 session_state 保存 GraphMemory 实例。
    """
    if "pipeline_memory" not in st.session_state:
        st.session_state["pipeline_memory"] = GraphMemory(max_len=20)

    mem = st.session_state["pipeline_memory"]
    if not isinstance(mem, GraphMemory):
        st.session_state["pipeline_memory"] = GraphMemory(max_len=20)
        mem = st.session_state["pipeline_memory"]

    return mem


# =========================
#  导出辅助（生成字节流）
# =========================


def build_word_bytes(result: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    export_to_word(result, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


def build_ppt_bytes(result: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
        tmp_path = tmp.name
    export_to_ppt(result, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


def build_excel_bytes(result: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    export_to_excel(result, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


# =========================
#  页面：教学流水线
# =========================


def page_teaching_pipeline():
    memory = get_web_memory()

    st.markdown(
        """
        <div class="top-title">
            <h1>📚 AI 教学流水线助手</h1>
            <div class="subtitle">
                从教学任务 → 教案 → PPT 大纲 → Excel 表，一键生成，还支持简单记忆。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 1])

    # -------- 左侧：输入表单 --------
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📝 填写本次教学任务</div>', unsafe_allow_html=True)

        with st.form("teaching_form"):
            course_title = st.text_input(
                "课程标题（必填）",
                value="人工智能入门：大模型基础",
                help="例如：一次函数基础、人工智能导论、Python 程序设计等",
            )
            student_level = st.text_input(
                "学生水平（必填）",
                value="高校大一新生",
                help="例如：初中二年级、高中一年级、高职一年级等",
            )

            col1, col2 = st.columns(2)
            with col1:
                duration = st.number_input(
                    "单节课时长（分钟）",
                    min_value=20,
                    max_value=180,
                    value=45,
                    step=5,
                )
            with col2:
                model = st.selectbox(
                    "使用模型",
                    options=["gpt-4.1-mini", "gpt-4.1"],
                    index=0,
                    help="如果你有 gpt-4.1 权限，也可以选它，质量略高但成本也更高。",
                )

            task = st.text_area(
                "本次教学任务描述（必填）",
                height=150,
                value=(
                    "帮助学生理解大语言模型（LLM）的基本概念，"
                    "知道它在日常生活和学习中的典型应用场景，"
                    "并通过一个简单示例体验与 AI 对话的过程。"
                ),
                help="建议用 2-5 句话描述清楚这节课最核心的任务。",
            )

            submitted = st.form_submit_button("🚀 生成教学设计")

        if submitted:
            if not course_title.strip() or not student_level.strip() or not task.strip():
                st.error("请先把『课程标题』『学生水平』『教学任务』填写完整。")
            else:
                try:
                    with st.spinner("正在调用大模型生成教学设计，请稍等……"):
                        result = run_teaching_pipeline(
                            task=task.strip(),
                            course_title=course_title.strip(),
                            student_level=student_level.strip(),
                            duration=int(duration),
                            memory=memory,
                            model=model,
                        )

                    st.success("教学设计生成完成 ✅")

                    # 把结果和导出文件缓存到 session_state
                    st.session_state["last_result"] = result

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state["export_word_bytes"] = build_word_bytes(result)
                    st.session_state["export_ppt_bytes"] = build_ppt_bytes(result)
                    st.session_state["export_excel_bytes"] = build_excel_bytes(result)
                    st.session_state["export_ts"] = ts

                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"发生未知错误：{e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # 导出按钮卡片
        if "last_result" in st.session_state:
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📂 导出与下载</div>', unsafe_allow_html=True)

            result = st.session_state["last_result"]
            ts = st.session_state.get("export_ts", datetime.now().strftime("%Y%m%d_%H%M%S"))
            course_title_safe = result.get("course_title", "teaching_plan").replace(" ", "_")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.download_button(
                    "⬇️ 下载 Word 教案",
                    data=st.session_state.get("export_word_bytes", b""),
                    file_name=f"{course_title_safe}_{ts}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            with col_b:
                st.download_button(
                    "⬇️ 下载 PPT 课件",
                    data=st.session_state.get("export_ppt_bytes", b""),
                    file_name=f"{course_title_safe}_{ts}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            with col_c:
                st.download_button(
                    "⬇️ 下载 Excel 结构表",
                    data=st.session_state.get("export_excel_bytes", b""),
                    file_name=f"{course_title_safe}_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.markdown(
                '<p class="small-hint">提示：建议先下载本地保存，再在 Word / PPT / Excel 中进行个性化微调。</p>',
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # -------- 右侧：结果展示 + 记忆 --------
    with col_right:
        # 结果展示
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">📖 本次 AI 教学设计结果</div>',
            unsafe_allow_html=True,
        )

        if "last_result" not in st.session_state:
            st.info("还没有结果。请先在左侧填写教学任务并点击「生成教学设计」。")
        else:
            result = st.session_state["last_result"]

            st.markdown(
                f"""
                <div class="small-hint">
                    <span class="result-tag">课程</span>{result.get("course_title", "")}<br/>
                    <span class="result-tag">学生水平</span>{result.get("student_level", "")}<br/>
                    <span class="result-tag">课时</span>{result.get("duration", "")} 分钟
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">一、教学目标</div>', unsafe_allow_html=True)
            st.markdown(result.get("goals", "（暂无）"))

            st.markdown('<div class="section-title">二、学情分析</div>', unsafe_allow_html=True)
            st.markdown(result.get("analysis", "（暂无）"))

            st.markdown('<div class="section-title">三、教学重点与难点</div>', unsafe_allow_html=True)
            st.markdown(result.get("key_points", "（暂无）"))

            st.markdown('<div class="section-title">四、教学过程设计</div>', unsafe_allow_html=True)
            st.markdown(result.get("process", "（暂无）"))

            st.markdown('<div class="section-title">五、课堂活动与问题设计</div>', unsafe_allow_html=True)
            st.markdown(result.get("activities", "（暂无）"))

            st.markdown('<div class="section-title">六、课后作业与延伸任务</div>', unsafe_allow_html=True)
            st.markdown(result.get("homework", "（暂无）"))

            st.markdown('<div class="section-title">七、教学反思建议</div>', unsafe_allow_html=True)
            st.markdown(result.get("reflection", "（暂无）"))

            with st.expander("查看原始教学设计全文（LLM 输出原文）", expanded=False):
                st.markdown(result.get("raw_plan_text", ""))

            with st.expander("查看自动生成的 PPT 大纲文本", expanded=False):
                st.markdown(result.get("ppt_outline", ""))

        st.markdown("</div>", unsafe_allow_html=True)

        # 内存 / 最近任务列表
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">🧠 最近教学任务记忆（简要）</div>',
            unsafe_allow_html=True,
        )

        mem = get_web_memory()
        if not mem.records:
            st.write("暂无历史记录。生成几次教学设计后，这里会显示最近的任务摘要。")
        else:
            for rec in reversed(mem.records[-5:]):
                with st.container():
                    st.markdown(
                        f"""
                        <div class="small-hint">
                            <span class="key-label">时间：</span>{rec.timestamp}<br/>
                            <span class="key-label">课程：</span>{rec.course_title}（{rec.student_level}，{rec.duration} 分钟）<br/>
                            <span class="key-label">任务摘要：</span>{rec.summary}
                        </div>
                        <hr style="margin: 0.4rem 0; opacity: 0.3;">
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
#  页面：关于 / 帮助
# =========================


def page_about():
    st.markdown(
        """
        <div class="top-title">
            <h1>ℹ️ 关于本应用</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        本应用由「AI 教学流水线」示例代码改造而成，核心功能：

        - 通过大模型自动生成标准化教学设计（教案）
        - 自动拆分为：教学目标 / 学情分析 / 教学过程 / 活动 / 作业 / 反思
        - 自动生成 PPT 大纲，并导出为 PPTX 文件
        - 自动导出 Word / Excel 结构表，便于后续微调与留档
        - 简单的 GraphMemory 记录最近的教学任务，方便回顾与分析

        使用方式建议：

        1. 作为「快速起稿工具」，老师先用本工具生成初稿；
        2. 再在 Word / PPT 中加入自己的教学经验和案例；
        3. 课堂后，可以把教学反思再输入回来，让 AI 帮你迭代下一轮教案。

        ⚙️ 配置说明（部署到 Streamlit Cloud）：

        - 在「Settings → Secrets」中添加：
          ```toml
          [default]
          OPENAI_API_KEY = "sk-xxxx"
          ```
        - 确保 `teaching_pipeline_cli.py` 和 `teaching_pipeline_web.py` 在同一仓库根目录；
        - `requirements.txt` 中包含：
          - streamlit
          - openai
          - python-docx
          - python-pptx
          - openpyxl

        """
    )


# =========================
#  主入口
# =========================


def main():
    set_page_style()

    with st.sidebar:
        st.title("📚 AI Teaching")
        page = st.radio("导航", ["教学流水线", "关于 / 帮助"], index=0)

    if page == "教学流水线":
        page_teaching_pipeline()
    else:
        page_about()


if __name__ == "__main__":
    main()
