# agent.py
# 这是一个“多工具 + 简单记忆”的示例智能体。
# 功能：
# 1）使用 OpenAI 大模型作为大脑（ChatOpenAI）
# 2）绑定多个工具（calculator, text_stats）
# 3）大模型自动决定是否调用工具、调用哪一个
# 4）Python 真正在本地执行工具，并把结果回传给大模型
# 5）在一次任务内，记录用过哪些工具、调用了几次，并在最后做一个简要总结

from typing import Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage


# 1. 定义第一个工具：计算器
@tool
def calculator(expression: str) -> str:
    """一个简单的计算器工具，输入算式（例如：'23 * 47'），返回计算结果。"""
    try:
        result = eval(expression)
        return f"表达式 {expression} 的计算结果是 {result}"
    except Exception as e:
        return f"计算出错：{e}"


# 2. 定义第二个工具：文本统计
@tool
def text_stats(text: str) -> str:
    """统计一段文本的长度（字符数）和大致单词数。"""
    char_count = len(text)
    # 对于英文：按空格粗略分词；中文因为没有空格，这里只是一个近似示意
    word_count = len(text.split())
    return f"这段文本共有 {char_count} 个字符，约 {word_count} 个词。"


def build_llm_with_tools():
    """
    构建一个带工具能力的 LLM：
    - 大脑：OpenAI Chat 模型（gpt-4.1-mini，仅示例，可按你账号情况调整）
    - 工具：calculator + text_stats
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,  # 稳定一点，减少胡说八道
    )

    # 把所有工具放进列表
    tools = [calculator, text_stats]

    # 让大模型“知道”自己有哪些工具可以用
    llm_with_tools = llm.bind_tools(tools)

    # 方便后面根据名字找到具体工具对象
    tool_map: Dict[str, any] = {t.name: t for t in tools}

    return llm_with_tools, tool_map


def run_agent(task: str) -> str:
    """
    运行一个“单轮任务，但允许多次、多工具调用”的 Agent 流程：

    1）构建带工具能力的 LLM & 工具映射表
    2）把任务作为 HumanMessage 发给模型
    3）模型如果发出 tool_calls：
        - 逐个在本地执行对应 Python 工具
        - 把工具结果包装成 ToolMessage 追加到消息列表
    4）最后再次调用模型，让它基于工具执行结果给出最终回答
    5）同时，我们在 Python 侧做一个简单“记忆”：记录本轮用了哪些工具、调用了几次
    """

    llm_with_tools, tool_map = build_llm_with_tools()

    # 简单“记忆容器”：记录本轮调用过的工具及相关信息
    tool_history: List[Tuple[str, dict, str]] = []

    # 对话消息列表，加入一个简单的 system 提示，告诉模型合理使用工具并用中文回答
    messages = [
        SystemMessage(
            content=(
                "你是一个会合理使用工具的中文智能体。"
                "当你需要精确计算或统计文本信息时，请优先调用提供给你的工具。"
                "最终回答时，请用通俗、清晰的中文解释你做了什么。"
            )
        ),
        HumanMessage(content=task),
    ]

    print("====== 发送任务给智能体 ======")
    print(task)
    print("================================\n")

    # 第一次调用：让模型决定是否要用工具
    response = llm_with_tools.invoke(messages)

    print("---- 模型第一次回复（可能包含工具调用） ----")
    print(response)
    print("----------------------------------------\n")

    # 如果没有工具调用，直接返回模型的文本回答
    if not getattr(response, "tool_calls", None):
        print("（模型没有调用任何工具，直接返回答案）\n")
        return str(response.content)

    # 有工具调用：先把模型这次的回复加入消息列表
    messages.append(response)

    # 逐个执行工具调用
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"模型决定调用工具：{tool_name}，参数：{tool_args}")

        if tool_name not in tool_map:
            tool_output = f"错误：未找到工具 {tool_name}"
        else:
            # 真正在本地执行对应工具
            tool_output = tool_map[tool_name].invoke(tool_args)

        print(f"工具 {tool_name} 执行结果：{tool_output}\n")

        # 记录到“记忆里”：工具名、参数、输出
        tool_history.append((tool_name, tool_args, str(tool_output)))

        # 把工具执行结果作为 ToolMessage 添加到消息列表中
        messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
            )
        )

    # 第二次调用：模型根据工具执行结果给出最终回答
    final_response = llm_with_tools.invoke(messages)

    print("---- 模型基于工具结果给出的最终回复 ----")
    print(final_response)
    print("----------------------------------------\n")

    # 整理“简单记忆总结”
    if tool_history:
        summary_lines = []
        summary_lines.append("本轮工具使用情况简要总结：")
        summary_lines.append(f"- 共调用工具 {len(tool_history)} 次。")

        # 统计每个工具调用次数
        stats: Dict[str, int] = {}
        for name, _, _ in tool_history:
            stats[name] = stats.get(name, 0) + 1

        for name, count in stats.items():
            summary_lines.append(f"  - 工具 `{name}` 被调用了 {count} 次。")

        summary_text = "\n".join(summary_lines)

        # 最终回答 = 模型的中文解释 + 工具使用总结
        return f"{final_response.content}\n\n{summary_text}"
    else:
        # 理论上不会走到这里，因为前面已经判断有 tool_calls
        return str(final_response.content)


def main():
    # 一个会用到两个工具的任务：
    # 1）用计算器精确计算 17 * 29
    # 2）用文本统计工具统计一句话的长度
    # 3）最后请模型用中文解释自己做了哪两步操作
    user_task = (
        "请先用你的计算器工具精确计算 17 * 29 的结果，"
        "然后用你的文本统计工具统计一下这句话有多少个字符："
        "『人工智能正在改变世界。』"
        "最后请用通俗中文总结，你分别做了哪两步操作。"
    )

    final_answer = run_agent(user_task)

    print("====== 智能体最终回答 ======")
    print(final_answer)
    print("================================")


if __name__ == "__main__":
    main()
