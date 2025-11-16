# graph_advanced_runtime.py
# 高级图式智能体运行时（教学场景版）：
# - Planner 节点：拆解任务
# - Worker 节点：按计划执行 + 调用工具（包括教学工具）
# - Critic 节点：审查 Worker 回答，决定是否重试
# - Error 节点：集中处理错误
# - Router：根据 state 决定下一步走哪个节点
#
# 这是一个迷你版的“工业级多智能体工作流”示例，已加入教学场景工具 generate_class_activities。

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
    # 加入新的教学工具 generate_class_activities
    tools = [calculator, text_stats, generate_class_activities]
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
#            节点实现
# ------------------------------

def planner_node(state: Dict[str, Any], planner_llm: ChatOpenAI) -> Dict[str, Any]:
    """Planner 节点：根据 user_input 生成分步计划"""
    user_input = state["user_input"]

    system_msg = SystemMessage(
        content=(
            "你是一个任务规划器（Planner）。"
            "你的目标是把用户任务拆分成 2-5 个清晰步骤。"
            "每一步用中文写清要做什么，可以提到是否需要使用 calculator、text_stats 或 generate_class_activities 等工具。"
            "只给步骤，不要直接给出最终答案。"
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
    - 使用工具完成任务
    - 写入 final_answer & tool_history
    """
    user_input = state["user_input"]
    planner_plan = state["planner_plan"]

    # 初始化 Worker 的系统提示和历史
    if not state["messages"]:
        base_system = (
            "你是执行智能体（Worker），擅长教学设计与课堂活动设计。"
            "以下是 Planner 提供的执行计划，请尽量按步骤完成任务：\n"
            f"{planner_plan}\n"
            "你可以调用这些工具：\n"
            "1）calculator：用于必要的数值计算；\n"
            "2）text_stats：用于统计文本长度；\n"
            "3）generate_class_activities：根据课程主题和年级生成 3 个课堂活动方案。\n"
            "在回答中，请用通俗中文说明你大致按哪些步骤完成了任务。"
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
    print(f"[Worker 节点] 第 {state['loop_count']} 次执行，准备调用执行器 LLM（第一次）")

    # 第一次调用：看是否要用工具
    try:
        response = exec_llm_with_tools.invoke(messages)
    except Exception as e:
        print("[Worker 错误] 第一次调用执行器失败：", e)
        state["error_msg"] = f"调用执行器失败：{e}"
        return state

    print("[Worker 节点] 已收到第一次回复")
    messages.append(response)
    print("[Worker 节点] 第一次回复：", response)

    # 工具调用
    tool_history: List[Tuple[str, dict, str]] = []

    if getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[Worker 节点] 模型调用工具：{tool_name}，参数：{tool_args}")

            if tool_name not in tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = tool_map[tool_name].invoke(tool_args)

            print(f"[Worker 节点] 工具输出：{tool_output}")

            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

    # 第二次调用：生成最终回答
    print("[Worker 节点] 准备调用执行器 LLM（第二次）")
    try:
        final_response = exec_llm_with_tools.invoke(messages)
    except Exception as e:
        print("[Worker 错误] 第二次调用执行器失败：", e)
        state["error_msg"] = f"第二次调用执行器失败：{e}"
        state["tool_history"] = tool_history
        return state

    print("[Worker 节点] 已收到第二次回复")
    messages.append(final_response)

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
            "你是一个审查智能体（Critic）。"
            "请检查 Worker 的回答是否："
            "1）逻辑合理；2）关键内容是否与用户需求一致；3）解释是否清楚、有条理。"
            "如果是教学设计任务，还要检查：教学目标是否具体、课堂活动是否可操作。\n"
            "请只输出两部分：\n"
            "第一行：0 或 1（0=无需重试，1=需要 Worker 重试并改进）\n"
            "从第二行开始：给出简短的审查理由。"
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
    print("====== 高级 Graph 智能体运行一次任务（教学场景版） ======")
    print("流程：Planner -> Worker -> Critic -> (可能重试) -> 结束/错误")
    print("=====================================================\n")

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
