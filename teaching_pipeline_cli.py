import os
from datetime import datetime
from typing import Dict, Any

# 尝试导入 streamlit（在本地 CLI 运行时可能不存在，不影响）
try:
    import streamlit as st
except ImportError:
    st = None

from openai import OpenAI


def _get_openai_api_key() -> str:
    """
    优先从环境变量中读取 OPENAI_API_KEY；
    如果在 Streamlit 环境中，则从 st.secrets 读取；
    如果都没有，给出清晰错误提示。
    """
    # 1. 先看环境变量（本地 CLI 使用）
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 2. 再看 Streamlit secrets（云端部署使用）
    if st is not None:
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            secret_key = None
        if secret_key:
            return secret_key

    # 3. 都没有就报错
    raise RuntimeError(
        "没有找到 OPENAI_API_KEY。\n"
        "本地运行时，请在系统环境变量中设置 OPENAI_API_KEY；\n"
        "Streamlit Cloud 上，请在 Settings → Secrets 中配置：\n"
        'OPENAI_API_KEY = "sk-xxxxxxxxxx"'
    )


# 全局 OpenAI 客户端
_client = OpenAI(api_key=_get_openai_api_key())

# 下面是你原来的其他 import / 代码，保持不变……
# 比如：
# from export_utils import export_to_word, export_to_ppt, export_to_excel
# class GraphMemory: ...
# def call_llm(...): ...


def call_llm(
    system_prompt: str,
    user_content: Union[str, Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    json_mode: bool = False,
    temperature: float = 0.4,
) -> str:
    """
    调用 OpenAI Chat Completions 的封装。
    json_mode=True 时，要求模型输出 JSON。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                user_content
                if isinstance(user_content, str)
                else json.dumps(user_content, ensure_ascii=False)
            ),
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


# ===================== 简单 GraphMemory =====================


class GraphMemory:
    """
    非常简单的“图记忆”：
    - 内部就是一个 list
    - 只保留最近 max_items 个条目
    - 提供 append() / to_list()
    """

    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self._items: List[Dict[str, Any]] = []

    def append(self, item: Dict[str, Any]) -> None:
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"GraphMemory(len={len(self._items)})"


def safe_memory_write(memory: Optional[GraphMemory], item: Dict[str, Any]) -> None:
    """
    安全写入记忆：
    - memory 为空则直接跳过
    - memory 有 append() 就写入
    """
    if memory is None:
        return
    if hasattr(memory, "append") and callable(getattr(memory, "append")):
        memory.append(item)
    else:
        print("GraphMemory 中未发现可用的写入方法，已跳过记忆写入。")


# ===================== 教学流水线（单次 LLM 调用） =====================


def run_teaching_pipeline(
    task: str,
    course_title: str,
    student_level: str,
    duration: str,
    memory: Optional[GraphMemory] = None,
) -> Dict[str, Any]:
    """
    一次 LLM 调用完成整套教学设计。

    返回 dict，包含：
      - goals
      - content_outline
      - activities
      - evaluation
      - ppt_structure
    """
    system_prompt = (
        "你是一名资深的教学设计专家，擅长为中小学、职教和高校老师设计教案。"
        "请根据用户给定的课程主题、学生对象和课时长度，输出结构化的教学设计结果。"
        "务必使用 JSON 格式输出，字段包括："
        "goals（教学目标，数组或带有knowledge/skills/attitude字段的对象）、"
        "content_outline（教学内容大纲，分节列出）、"
        "activities（课堂活动设计，至少3个活动，含目标、步骤、时间、材料）、"
        "evaluation（评价方式和作业）、"
        "ppt_structure（PPT 结构，为数组，每一页有 title 和 bullets 数组）。"
    )

    user_payload = {
        "task": task,
        "course_title": course_title,
        "student_level": student_level,
        "duration": duration,
    }

    raw = call_llm(system_prompt, user_payload, json_mode=True)
    try:
        data = json.loads(raw)
    except Exception:
        # 极端情况兜底：模型没按 JSON 格式来
        data = {
            "goals": raw,
            "content_outline": "",
            "activities": "",
            "evaluation": "",
            "ppt_structure": [],
        }

    if "ppt_structure" not in data or not isinstance(data.get("ppt_structure"), list):
        data["ppt_structure"] = []

    result: Dict[str, Any] = {
        "task": task,
        "course_title": course_title,
        "student_level": student_level,
        "duration": duration,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "goals": data.get("goals", ""),
        "content_outline": data.get("content_outline", ""),
        "activities": data.get("activities", ""),
        "evaluation": data.get("evaluation", ""),
        "ppt_structure": data.get("ppt_structure", []),
        "raw_json": data,
    }

    safe_memory_write(
        memory,
        {
            "timestamp": result["generated_at"],
            "task": task,
            "course_title": course_title,
            "student_level": student_level,
            "duration": duration,
            "summary": {
                "has_goals": bool(result["goals"]),
                "has_activities": bool(result["activities"]),
                "ppt_pages": len(result["ppt_structure"]),
            },
        },
    )

    return result


# ===================== 导出 Word / PPT =====================


def _ensure_document() -> Any:
    if Document is None:
        raise RuntimeError("python-docx 未安装，请先运行: pip install python-docx")
    return Document


def _ensure_presentation() -> Any:
    if Presentation is None:
        raise RuntimeError("python-pptx 未安装，请先运行: pip install python-pptx")
    return Presentation


def export_to_word(
    result: Dict[str, Any],
    filename: Optional[str] = None,
    *,
    to_file: bool = True,
) -> Union[str, bytes]:
    """
    由流水线结果生成 Word 文件。
    """
    DocumentCls = _ensure_document()
    doc = DocumentCls()

    doc.add_heading(result.get("course_title", "教学设计"), level=0)

    meta_p = doc.add_paragraph()
    meta_p.add_run(f"任务：{result.get('task', '')}\n").bold = True
    meta_p.add_run(
        f"学生对象：{result.get('student_level', '')}  "
        f"课时长度：{result.get('duration', '')}\n"
    )
    meta_p.add_run(f"生成时间：{result.get('generated_at', '')}")

    # 一、教学目标
    doc.add_heading("一、教学目标", level=1)
    goals = result.get("goals", "")
    if isinstance(goals, dict):
        for key, val in goals.items():
            doc.add_paragraph(f"{key}：{val}", style="List Bullet")
    elif isinstance(goals, list):
        for g in goals:
            doc.add_paragraph(str(g), style="List Bullet")
    else:
        doc.add_paragraph(str(goals))

    # 二、教学内容大纲
    doc.add_heading("二、教学内容大纲", level=1)
    content = result.get("content_outline", "")
    if isinstance(content, list):
        for sec in content:
            if isinstance(sec, dict):
                title = sec.get("title") or sec.get("name") or "模块"
                doc.add_heading(title, level=2)
                points = sec.get("points") or sec.get("bullets") or []
                for p in points:
                    doc.add_paragraph(str(p), style="List Bullet")
            else:
                doc.add_paragraph(str(sec), style="List Bullet")
    else:
        doc.add_paragraph(str(content))

    # 三、课堂活动
    doc.add_heading("三、课堂活动设计", level=1)
    acts = result.get("activities", "")
    if isinstance(acts, list):
        for idx, act in enumerate(acts, start=1):
            if isinstance(act, dict):
                doc.add_heading(f"活动{idx}：{act.get('name','')}", level=2)
                for field in ["goal", "steps", "time", "materials"]:
                    if field in act:
                        doc.add_paragraph(f"{field}：{act[field]}")
            else:
                doc.add_paragraph(str(act))
    else:
        doc.add_paragraph(str(acts))

    # 四、评价方式
    doc.add_heading("四、评价方式与作业", level=1)
    eva = result.get("evaluation", "")
    if isinstance(eva, list):
        for e in eva:
            doc.add_paragraph(str(e), style="List Bullet")
    else:
        doc.add_paragraph(str(eva))

    buf = BytesIO()
    doc.save(buf)
    data = buf.getvalue()

    if to_file:
        if not filename:
            filename = "teaching_plan.docx"
        with open(filename, "wb") as f:
            f.write(data)
        return filename
    else:
        return data


def export_to_ppt(
    result: Dict[str, Any],
    filename: Optional[str] = None,
    *,
    to_file: bool = True,
) -> Union[str, bytes]:
    """
    由流水线结果生成 PPTX 文件。
    ppt_structure 期望为 [{title: str, bullets: [str]}] 列表，
    若没有则自动生成一个通用结构。
    """
    PresentationCls = _ensure_presentation()
    prs = PresentationCls()

    ppt_structure = result.get("ppt_structure") or []

    if not ppt_structure:
        # 兜底：简单 6 页结构
        ppt_structure = [
            {
                "title": "课程介绍与教学目标",
                "bullets": [result.get("course_title", ""), str(result.get("student_level", ""))],
            },
            {"title": "教学内容大纲", "bullets": ["主要模块与知识点概览"]},
            {"title": "重点知识讲解", "bullets": ["核心概念", "关键原理"]},
            {"title": "课堂活动设计", "bullets": ["活动1", "活动2", "活动3"]},
            {"title": "评价方式与作业", "bullets": ["过程性评价", "终结性评价"]},
            {"title": "课堂小结与拓展", "bullets": ["本课小结", "课后拓展建议"]},
        ]

    # 删除默认空白第一页
    while len(prs.slides) > 0:
        r_id = prs.slides._sldIdLst[0].rId  # type: ignore[attr-defined]
        prs.part.drop_rel(r_id)  # type: ignore[attr-defined]
        del prs.slides._sldIdLst[0]  # type: ignore[attr-defined]

    title_layout = prs.slide_layouts[0]   # 标题
    bullet_layout = prs.slide_layouts[1]  # 标题 + 内容

    # 第 1 页：课程标题
    slide = prs.slides.add_slide(title_layout)
    slide.shapes.title.text = result.get("course_title", "教学设计")
    subtitle = slide.placeholders[1]
    subtitle.text = (
        f"{result.get('student_level','')}  |  时长：{result.get('duration','')}"
    )

    # 后续页
    for page in ppt_structure:
        title = ""
        bullets: List[str] = []
        if isinstance(page, dict):
            title = str(page.get("title", ""))
            b = page.get("bullets") or page.get("points") or []
            if isinstance(b, list):
                bullets = [str(x) for x in b]
            else:
                bullets = [str(b)]
        else:
            title = str(page)
            bullets = []

        slide = prs.slides.add_slide(bullet_layout)
        slide.shapes.title.text = title or "教学内容"
        body = slide.placeholders[1].text_frame
        body.clear()

        if bullets:
            # 第一个项目直接赋值到 text_frame.text
            body.text = bullets[0]
            # 其余项目用 add_paragraph()，再赋 p.text
            for b in bullets[1:]:
                p = body.add_paragraph()
                p.text = b
        else:
            body.text = ""

    buf = BytesIO()
    prs.save(buf)
    data = buf.getvalue()

    if to_file:
        if not filename:
            filename = "teaching_ppt.pptx"
        with open(filename, "wb") as f:
            f.write(data)
        return filename
    else:
        return data


# ===================== CLI 入口 =====================


def pretty_print_result(result: Dict[str, Any]) -> None:
    print("\n========== 教学流水线结果摘要 ==========")
    print(
        f"课程：{result.get('course_title','')}  | "
        f"对象：{result.get('student_level','')}  | "
        f"时长：{result.get('duration','')}"
    )
    print(f"生成时间：{result.get('generated_at','')}")
    print("\n[教学目标]：")
    print(result.get("goals", ""))
    print("\n[教学内容大纲]：")
    print(result.get("content_outline", ""))
    print("\n[课堂活动设计]：")
    print(result.get("activities", ""))
    print("\n[评价方式]：")
    print(result.get("evaluation", ""))
    print("=======================================\n")


def main() -> None:
    print("====== 教学流水线 CLI ======")
    print("本程序会调用 OpenAI 接口，请确保已设置 OPENAI_API_KEY 环境变量。\n")

    memory = GraphMemory(max_items=10)

    while True:
        task = input("请输入一个教学任务（或输入 退出 结束）：").strip()
        if task in {"退出", "exit", "quit", "q"}:
            break
        if not task:
            continue

        course_title = input("课程名称：").strip() or "人工智能基础入门"
        student_level = input("学生对象（如：中职一年级、大一）：").strip() or "中职一年级"
        duration = input("课时长度（如：1 课时 / 90 分钟）：").strip() or "1 课时"

        print("\n[系统] 正在生成教学设计，请稍等...")
        try:
            result = run_teaching_pipeline(
                task, course_title, student_level, duration, memory
            )
        except Exception as e:
            print(f"生成教学设计时出错：{e}")
            continue

        pretty_print_result(result)

        ans = input("是否导出为 Word 教案？(y/n)：").strip().lower()
        if ans == "y":
            try:
                path = export_to_word(
                    result, filename="teaching_plan.docx", to_file=True
                )
                print(f"已导出 Word 教案：{path}")
            except Exception as e:
                print(f"导出 Word 失败：{e}")

        ans = input("是否导出为 PPT 结构？(y/n)：").strip().lower()
        if ans == "y":
            try:
                path = export_to_ppt(
                    result, filename="teaching_ppt.pptx", to_file=True
                )
                print(f"已导出 PPT：{path}")
            except Exception as e:
                print(f"导出 PPT 失败：{e}")

        print("\n[记忆状态] 最近任务数量：", len(memory))
        print("============================================\n")


if __name__ == "__main__":
    main()
