# graph_style_runtime.py
# 用“图式思维”重写 Planner + Worker 智能体
# 目标：用最简单的 Python 代码，把“状态 State + 节点 Node + 流程 Graph”跑起来

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
    tools = [calculator, text_stats]
    llm_with_tools = llm.bind_tools(tools)
    tool_map: Dict[str, Any] = {t.name: t for t in tools}
    return llm_with_tools, tool_map


# ------------------------------
#        状态 State 结构
# ------------------------------
# 我们用一个 dict 来保存“整张图的状态”

def init_state(user_input: str) -> Dict[str, Any]:
    """
    初始化图的状态：
    - user_input: 用户当前输入
    - planner_plan: 规划器输出的步骤
    - messages: Worker 对话历史
    - tool_history: 工具调用记录
    - final_answer: 最终回答
    - finished: 图是否已经结束
    """
    state: Dict[str, Any] = {
        "user_input": user_input,
        "planner_plan": "",
        "messages": [],         # Worker 的对话历史
        "tool_history": [],     # 每轮工具调用记录
        "final_answer": "",
        "finished": False,
    }
    return state


# ------------------------------
#            节点 Node
# ------------------------------

def planner_node(state: Dict[str, Any], planner_llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Planner 节点：
    输入：state["user_input"]
    输出：state["planner_plan"]
    """
    user_input = state["user_input"]

    system_msg = SystemMessage(
        content=(
            "你是一个任务规划器（Planner）。"
            "你的目标是把用户任务拆成 2-5 个清晰步骤。"
            "每一步用中文写清要做什么，可以提到是否需要使用 calculator 或 text_stats 工具。"
            "只给步骤，不要直接给出最终答案。"
        )
    )
    user_msg = HumanMessage(content=f"用户的需求是：{user_input}。请给出分步计划。")

    print("\n[Graph][Planner 节点] 正在生成计划...")
    try:
        response = planner_llm.invoke([system_msg, user_msg])
    except Exception as e:
        print("[Graph][Planner 错误] 调用规划器失败：", e)
        plan_text = (
            "步骤1：尝试直接使用可用工具或语言能力解决问题。\n"
            "步骤2：如果无法解决，向用户说明原因。"
        )
    else:
        plan_text = str(response.content)

    print("[Graph][Planner 节点] 计划为：\n", plan_text, "\n")

    state["planner_plan"] = plan_text
    return state


def worker_node(state: Dict[str, Any],
                exec_llm_with_tools,
                tool_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker 节点：
    输入：
      - state["user_input"]
      - state["planner_plan"]
    行为：
      - 构造 Worker 自己的 messages（包含计划 + 用户输入）
      - 调用执行器 LLM + 工具
      - 写入：
          state["messages"]
          state["tool_history"]
          state["final_answer"]
    """
    user_input = state["user_input"]
    planner_plan = state["planner_plan"]

    # 如果还没有 Worker 的历史，就初始化
    if not state["messages"]:
        state["messages"] = [
            SystemMessage(
                content=(
                    "你是执行智能体（Worker）。"
                    "以下是 Planner 提供的执行计划，请尽量按步骤完成任务：\n"
                    f"{planner_plan}\n"
                    "你可以调用 calculator 和 text_stats 工具。"
                    "最终回答时，请用通俗中文说明你做了什么。"
                )
            )
        ]

    messages: List[BaseMessage] = state["messages"]

    # 加入本轮用户输入
    messages.append(HumanMessage(content=user_input))

    print("[Graph][Worker 节点] 准备调用执行器 LLM（第一次）")
    try:
        response = exec_llm_with_tools.invoke(messages)
    except Exception as e:
        print("[Graph][Worker 错误] 第一次调用执行器失败：", e)
        state["final_answer"] = "执行任务时调用大模型失败，可能是网络或限流问题。"
        state["finished"] = True
        return state

    print("[Graph][Worker 节点] 已收到第一次回复")
    messages.append(response)
    print("[Graph][Worker 节点] 第一次回复：", response)

    # 工具调用
    tool_history: List[Tuple[str, dict, str]] = []

    if getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[Graph][Worker 节点] 模型调用工具：{tool_name}，参数：{tool_args}")

            if tool_name not in tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = tool_map[tool_name].invoke(tool_args)

            print(f"[Graph][Worker 节点] 工具输出：{tool_output}")

            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

    # 第二次调用，生成最终回答
    print("[Graph][Worker 节点] 准备调用执行器 LLM（第二次）")
    try:
        final_response = exec_llm_with_tools.invoke(messages)
    except Exception as e:
        print("[Graph][Worker 错误] 第二次调用执行器失败：", e)
        state["final_answer"] = "第二次调用执行器失败，可能是网络或限流问题。"
        state["tool_history"] = tool_history
        state["finished"] = True
        return state

    print("[Graph][Worker 节点] 已收到第二次回复")
    messages.append(final_response)

    state["messages"] = messages
    state["tool_history"] = tool_history
    state["final_answer"] = str(final_response.content)
    state["finished"] = True

    return state


# ------------------------------
#          简易 Graph 运行器
# ------------------------------

def run_graph(user_input: str):
    """
    最小图：
      State --(planner_node)--> State --(worker_node)--> State(finished=True)
    """

    # 1. 初始化状态
    state = init_state(user_input)

    # 2. 初始化大模型
    planner_llm = build_planner_llm()
    exec_llm_with_tools, tool_map = build_exec_llm_with_tools()

    # 3. Planner 节点
    state = planner_node(state, planner_llm)

    # 4. Worker 节点
    state = worker_node(state, exec_llm_with_tools, tool_map)

    # 5. 输出整体结果
    print("\n========== 图执行结束 ==========")
    print("\n[Planner 规划结果]")
    print(state["planner_plan"])

    print("\n[Worker 最终回答]")
    print(state["final_answer"])

    if state["tool_history"]:
        print("\n[工具调用记录]")
        for name, args, out in state["tool_history"]:
            print(f"- {name} 参数={args}, 输出={out}")

    print("\n===============================")


# ------------------------------
#              主程序
# ------------------------------

def main():
    print("====== Graph 风格运行一次任务 ======")
    print("示例：多步任务由 Planner 拆解，再由 Worker + 工具执行")
    print("===================================\n")

    while True:
        user_input = input("请输入一个任务（输入 退出 结束）：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束。")
            break

        if not user_input:
            continue

        run_graph(user_input)
        print("\n-----------------------------------\n")


if __name__ == "__main__":
    main()
