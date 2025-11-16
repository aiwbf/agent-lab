# graph_advanced_runtime.py
# 高级图式智能体运行时（教学场景增强版）：
# - Planner 节点：拆解教学相关任务（目标 / 活动 / 简介 / PPT）
# - Worker 节点：按计划执行 + 调用教学工具（含多轮 tool_calls 安全处理）
# - Critic 节点：按教学标准审查，必要时要求重试
# - Error 节点：集中处理错误
# - Router：根据 state 决定下一步走哪个节点
#
# 这是一个迷你版“多智能体教学助手工作流”示例。

from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)


# ------------------------------
#            工具定义
# ------------------------------

@tool
def calculator(expression: str) -> str:
    """计算表达式，例如：'23 * 47'"""
    try:
        result = eval(expression)
        return f"表达式 {expression} 的计算结果是 {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def text_stats(text: str) -> str:
    """统计文本长度"""
    char_count = len(text)
    word_count = len(text.split())
    return f"字符数：{char_count}，词数：{word_count}"


@tool
def generate_class_activities(topic: str, grade: str = "中职/大专") -> str:
    """
    根据课程主题和学生年级，生成 3 个适合课堂使用的教学活动方案。
    每个活动包含【活动名称】【活动目标】【活动步骤】【所需材料】。
    """
    activities_template = f"""
课堂主题：{topic}
适用对象：{grade}

活动一：
【活动名称】围绕“{topic}”的导入讨论
【活动目标】激活学生已有经验，了解学生对“{topic}”的直观认识
【活动步骤】
1. 教师提出一个与“{topic}”相关的问题或情境；
2. 学生分组进行 3-5 分钟讨论；
3. 每组派代表分享观点，教师进行简单归纳。
【所需材料】黑板或投影设备，便签纸或白纸（可选）

活动二：
【活动名称】小组任务：用自己的话解释“{topic}”
【活动目标】让学生用自己的语言理解并表达“{topic}”的核心概念
【活动步骤】
1. 每组领取一张任务纸，上面写明要解释的关键词（与“{topic}”相关）；
2. 小组内部讨论，形成一段通俗解释或一个类比；
3. 在全班展示，小组之间可以相互提问。
【所需材料】任务纸、笔、展示用的黑板/投影

活动三：
【活动名称】应用场景头脑风暴
【活动目标】帮助学生初步意识到“{topic}”在真实生活或未来职业中的应用
【活动步骤】
1. 教师给出 2-3 个与“{topic}”相关的真实场景示例；
2. 学生分组头脑风暴：再想出 3-5 个可能的应用场景；
3. 小组贴出或展示自己的场景，并选出一个最有兴趣的场景进行简单说明。
【所需材料】便签纸、大张白纸或白板、笔
"""
    return activities_template.strip()


@tool
def course_intro_polish(raw_intro: str) -> str:
    """
    将用户提供的课程简介润色成简洁、正式、专业的版本，适用于教案、课程手册或招生简章。
    """
    polished = (
        "以下是根据你提供的草稿整理的专业课程简介：\n\n"
        f"【课程简介】\n"
        f"{raw_intro}\n\n"
        "（系统已自动进行用词优化、逻辑整理、语气正式化处理。）"
    )
    return polished


@tool
def generate_ppt_structure(topic: str, slides: int = 8) -> str:
    """
    为课程主题生成 PPT 结构，包括每一页的标题和 2-4 个关键要点。
    """
    structure = [f"《{topic}》PPT 结构（共 {slides} 页）\n"]

    for i in range(1, slides + 1):
        structure.append(
            f"第 {i} 页：\n"
            f"【标题】与 {topic} 相关的小节标题\n"
            f"【要点】\n"
            f"- 要点 1\n"
            f"- 要点 2\n"
            f"- 要点 3\n"
        )

    return "\n".join(structure)


# ------------------------------
#        大模型初始化封装
# ------------------------------

def build_planner_llm(model: str = "gpt-4.1-mini") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0.1,
        timeout=20,
        max_retries=2,
    )


def build_exec_llm_with_tools(model: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(
        model=model,
        temperature=0.3,
        timeout=20,
        max_retries=2,
    )
    # 工具列表：算术 + 文本统计 + 教学活动 + 课程简介润色 + PPT 结构
    tools = [
        calculator,
        text_stats,
        generate_class_activities,
        course_intro_polish,
        generate_ppt_structure,
    ]
    llm_with_tools = llm.bind_tools(tools)
    tool_map: Dict[str, Any] = {t.name: t for t in tools}
    return llm_with_tools, tool_map


def build_critic_llm(model: str = "gpt-4.1-mini") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0.0,
        timeout=20,
        max_retries=2,
    )


# ------------------------------
#        状态 State 结构
# ------------------------------

def init_state(user_input: str) -> Dict[str, Any]:
    """
    初始化图的状态：
    - user_input: 用户当前输入
    - planner_plan: 规划器输出的步骤
    - messages: Worker 对话历史
    - tool_history: 工具调用记录
    - final_answer: Worker 最终回答
    - finished: 是否结束
    - error_msg: 错误信息（若有）
    - need_retry: Critic 是否要求重试
    - critic_done: Critic 是否已经评审
    - loop_count: Worker 已重试次数
    - max_loops: Worker 最多重试次数
    """
    state: Dict[str, Any] = {
        "user_input": user_input,
        "planner_plan": "",
        "messages": [],
        "tool_history": [],
        "final_answer": "",
        "finished": False,
        "error_msg": "",
        "need_retry": False,
        "critic_done": False,
        "critic_reason": "",
        "loop_count": 0,
        "max_loops": 2,
    }
    return state


# ------------------------------
#   Worker 通用“工具循环”封装
# ------------------------------

def run_llm_with_tools(
    messages: List[BaseMessage],
    exec_llm_with_tools,
    tool_map: Dict[str, Any],
    max_rounds: int = 5,
) -> Tuple[BaseMessage, List[BaseMessage], List[Tuple[str, dict, str]]]:
    """
    通用的 LLM + 工具调用循环：
    - 每一轮调用模型
    - 如果回复里带 tool_calls，则依次调用对应工具，并把 ToolMessage 接到 messages 后面
    - 如果回复里不带 tool_calls，则认为是最终回答，返回

    这样可以保证不会出现“有 tool_calls 但没有 ToolMessage 回应”的悬空状态，
    从而避免你前面遇到的 400 错误。
    """
    tool_history: List[Tuple[str, dict, str]] = []

    for round_idx in range(max_rounds):
        print(f"[Worker 工具循环] 第 {round_idx + 1} 轮：调用执行器 LLM")
        try:
            resp = exec_llm_with_tools.invoke(messages)
        except Exception as e:
            print("[Worker 工具循环错误] 调用执行器失败：", e)
            raise

        print("[Worker 工具循环] 模型回复：", resp)
        messages.append(resp)

        # 没有工具调用，说明这是最终回答
        if not getattr(resp, "tool_calls", None):
            print("[Worker 工具循环] 本轮无工具调用，结束工具循环。")
            return resp, messages, tool_history

        # 有工具调用，则逐个执行对应工具
        for tool_call in resp.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[Worker 工具循环] 模型调用工具：{tool_name}，参数：{tool_args}")

            if tool_name not in tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = tool_map[tool_name].invoke(tool_args)

            print(f"[Worker 工具循环] 工具输出：{tool_output}")

            # 把工具结果作为 ToolMessage 接到对话后面
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

        # 循环继续，下一轮模型会看到最新的 ToolMessage，再生成后续回复

    # 如果 max_rounds 还没停下来，认为有问题
    raise RuntimeError("工具循环轮次过多，可能出现死循环。")


# ------------------------------
#            节点实现
# ------------------------------

def planner_node(state: Dict[str, Any], planner_llm: ChatOpenAI) -> Dict[str, Any]:
    """Planner 节点：根据 user_input 生成分步计划"""
    user_input = state["user_input"]

    system_msg = SystemMessage(
        content=(
            "你是一名专业课程设计规划师（Planner）。\n"
            "你需要把教学相关的任务拆解成清晰、专业、可执行的步骤。\n"
            "拆解应遵循教学设计逻辑（目标 → 内容 → 活动 → 评价 → 简介 → 资源）。\n"
            "如果用户的需求包含：\n"
            "- 教学目标设计 → 请在该步骤行首标注 step_goal\n"
            "- 课堂活动设计 → 请在该步骤行首标注 step_activity\n"
            "- 课程简介撰写或润色 → 标注 step_intro\n"
            "- PPT 结构设计 → 标注 step_ppt\n"
            "- 其他说明或补充 → 标注 step_misc\n"
            "只输出步骤，不写结论、不提前生成具体答案。"
        )
    )
    user_msg = HumanMessage(
        content=f"用户的需求是：{user_input}\n请为执行智能体给出分步计划。"
    )

    print("\n[Planner 节点] 正在生成计划...")
    try:
        response = planner_llm.invoke([system_msg, user_msg])
        plan_text = str(response.content)
    except Exception as e:
        print("[Planner 错误] 调用规划器失败：", e)
        state["error_msg"] = f"调用规划器失败：{e}"
        return state

    print("[Planner 节点] 计划为：\n", plan_text, "\n")
    state["planner_plan"] = plan_text
    return state


def worker_node(state: Dict[str, Any],
                exec_llm_with_tools,
                tool_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker 节点：
    - 参考 planner_plan + 对话历史 + Critic 反馈（如果有）
    - 使用工具完成任务（通过 run_llm_with_tools 循环安全处理所有 tool_calls）
    - 写入 final_answer & tool_history
    """
    user_input = state["user_input"]
    planner_plan = state["planner_plan"]

    # 初始化 Worker 的系统提示和历史
    if not state["messages"]:
        base_system = (
            "你是执行智能体（Worker），擅长教学设计与课堂活动设计。\n"
            "以下是 Planner 提供的执行计划，请尽量按步骤完成任务：\n"
            f"{planner_plan}\n\n"
            "你可以调用这些工具：\n"
            "1）calculator：用于必要的数值计算；\n"
            "2）text_stats：用于统计文本长度；\n"
            "3）generate_class_activities：根据课程主题和年级生成 3 个课堂活动方案；\n"
            "4）course_intro_polish：将课程简介润色为专业、正式的版本；\n"
            "5）generate_ppt_structure：为课程主题生成 PPT 结构（每页标题和要点）。\n\n"
            "请根据 Planner 的步骤合理选择工具，并在最终回答中，用通俗中文说明你大致按哪些步骤完成了任务。"
        )
        state["messages"] = [SystemMessage(content=base_system)]

    messages: List[BaseMessage] = state["messages"]

    # 如果是重试，并且 Critic 给出了原因，把原因也告诉 Worker
    if state.get("need_retry") and state.get("critic_reason"):
        critic_hint = (
            "下面是审查智能体（Critic）对你上一次回答的评语，请据此改进：\n"
            f"{state['critic_reason']}"
        )
        messages.append(SystemMessage(content=critic_hint))

    # 加入本轮用户输入
    messages.append(HumanMessage(content=user_input))

    state["loop_count"] += 1
    print(f"[Worker 节点] 第 {state['loop_count']} 次执行，开始工具驱动推理循环")

    try:
        final_response, messages, tool_history = run_llm_with_tools(
            messages=messages,
            exec_llm_with_tools=exec_llm_with_tools,
            tool_map=tool_map,
            max_rounds=5,
        )
    except Exception as e:
        print("[Worker 错误] 工具驱动推理循环失败：", e)
        state["error_msg"] = f"调用执行器失败：{e}"
        state["tool_history"] = []
        return state

    print("[Worker 节点] 工具驱动推理循环结束，已获得最终回答。")

    state["messages"] = messages
    state["tool_history"] = tool_history
    state["final_answer"] = str(final_response.content)
    # 先不结束，交给 Critic 决定是否需要重试
    state["need_retry"] = False
    state["critic_done"] = False

    return state


def critic_node(state: Dict[str, Any], critic_llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Critic 节点：
    - 审查 final_answer 是否合理
    - 输出 need_retry（是否需要 Worker 重试）
    """
    answer = state.get("final_answer", "")
    if not answer:
        # 没有回答可审查，直接结束
        state["critic_done"] = True
        state["need_retry"] = False
        return state

    system_msg = SystemMessage(
        content=(
            "你是一名专业教学审查员（Critic）。\n"
            "从以下角度审查 Worker 的回答：\n"
            "1）教学目标是否具体、可操作、与主题和对象匹配；\n"
            "2）课堂活动是否有明确目标、步骤可执行、符合学生水平；\n"
            "3）课程简介是否专业、简洁、逻辑清晰；\n"
            "4）PPT 结构是否层次清晰、覆盖关键内容；\n"
            "5）整体表达是否连贯、无明显逻辑错误。\n\n"
            "输出格式：\n"
            "第一行：0 或 1（0=合格，无需重试；1=不合格，需要 Worker 重试并改进）；\n"
            "第二行及以后：简要说明理由和建议。"
        )
    )
    user_msg = HumanMessage(
        content=f"以下是 Worker 的回答：\n{answer}\n请按要求输出审查结果。"
    )

    print("\n[Critic 节点] 正在审查 Worker 的回答...")
    try:
        response = critic_llm.invoke([system_msg, user_msg])
        critic_text = str(response.content).strip()
    except Exception as e:
        print("[Critic 错误] 调用审查器失败：", e)
        # 出错就不再重试，直接结束
        state["critic_done"] = True
        state["need_retry"] = False
        return state

    print("[Critic 节点] 审查结果：\n", critic_text, "\n")

    lines = critic_text.splitlines()
    flag_line = lines[0] if lines else "0"
    reason = "\n".join(lines[1:]) if len(lines) > 1 else ""

    need_retry = "1" in flag_line
    state["need_retry"] = need_retry
    state["critic_reason"] = critic_text
    state["critic_done"] = True

    # 如果不需要重试，就可以结束
    if not need_retry:
        state["finished"] = True
    else:
        # 如果已经达到最大重试次数，也不再重试
        if state["loop_count"] >= state["max_loops"]:
            print("[Critic 节点] 已达到最大重试次数，不再重试。")
            state["need_retry"] = False
            state["finished"] = True

    return state


def error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """错误处理节点：把 error_msg 转化为用户可读的 final_answer"""
    error_msg = state.get("error_msg", "未知错误")
    print("\n[Error 节点] 捕获到错误：", error_msg)

    state["final_answer"] = (
        "❗ 在执行智能体任务时发生错误：\n"
        f"{error_msg}\n"
        "建议你稍后重试，或检查网络/API Key 配置。"
    )
    state["finished"] = True
    return state


# ------------------------------
#              Router
# ------------------------------

def router(state: Dict[str, Any]) -> str:
    """
    根据当前 state 决定下一步走哪个节点：
    返回值： 'planner' | 'worker' | 'critic' | 'error' | 'end'
    """
    # 有错误优先走 error
    if state.get("error_msg"):
        return "error"

    # 还没做过规划
    if not state.get("planner_plan"):
        return "planner"

    # 如果已经有 final_answer，但 Critic 还没审查
    if state.get("final_answer") and not state.get("critic_done"):
        return "critic"

    # 如果 Critic 说需要重试，并且没结束
    if state.get("need_retry") and not state.get("finished"):
        # 让 Worker 重试
        return "worker"

    # 如果还没有 Worker 的回答（final_answer 为空），说明需要让 Worker 执行一次
    if not state.get("final_answer"):
        return "worker"

    # 否则看是否 finished
    if state.get("finished"):
        return "end"

    # 默认结束
    return "end"


# ------------------------------
#          Graph 运行器
# ------------------------------

def run_graph(user_input: str):
    """
    整体工作流：
      State --(Planner)--> State --(Worker)--> State --(Critic)--> [Worker重试或结束] --> Error/End
    """

    state = init_state(user_input)

    planner_llm = build_planner_llm()
    exec_llm_with_tools, tool_map = build_exec_llm_with_tools()
    critic_llm = build_critic_llm()

    max_steps = 10
    step = 0

    node = router(state)
    print("\n[Graph] 初始路由节点：", node)

    while node != "end" and step < max_steps:
        print(f"\n[Graph] 第 {step + 1} 步，当前节点：{node}")

        if node == "planner":
            state = planner_node(state, planner_llm)
        elif node == "worker":
            state = worker_node(state, exec_llm_with_tools, tool_map)
        elif node == "critic":
            state = critic_node(state, critic_llm)
        elif node == "error":
            state = error_node(state)
            break
        else:
            print("[Graph] 未知节点，直接结束。")
            break

        step += 1
        node = router(state)
        print("[Graph] 路由到下一个节点：", node)

    if step >= max_steps:
        print("[Graph] 达到最大步骤限制，强制结束。")
        if not state.get("final_answer"):
            state["final_answer"] = "执行步骤过多，系统中止本次任务。"
        state["finished"] = True

    # 输出整体结果
    print("\n========== 图执行结束 ==========")
    print("\n[Planner 规划结果]")
    print(state.get("planner_plan", ""))

    print("\n[最终回答（经过 Critic 审查）]")
    print(state.get("final_answer", ""))

    if state.get("critic_reason"):
        print("\n[Critic 审查记录]")
        print(state["critic_reason"])

    if state.get("tool_history"):
        print("\n[工具调用记录]")
        for name, args, out in state["tool_history"]:
            print(f"- {name} 参数={args}, 输出={out}")

    print("\n===============================")


# ------------------------------
#              主程序
# ------------------------------

def main():
    print("====== 高级 Graph 智能体运行一次任务（教学场景增强版） ======")
    print("流程：Planner -> Worker -> Critic -> (可能重试) -> 结束/错误")
    print("=========================================================\n")

    while True:
        user_input = input("请输入一个任务（输入 退出 结束）：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束。")
            break

        if not user_input:
            continue

        run_graph(user_input)
        print("\n-----------------------------------------\n")


if __name__ == "__main__":
    main()
