# agent_chat.py
# 一个支持“多轮对话 + 多工具 + 简单记忆（会话级）”的智能体。
#
# 能力：
# 1）你可以反复输入问题，它会记住之前聊过的内容（简单会话记忆）。
# 2）有两个工具：
#    - calculator：计算表达式，例如 "23 * 47"
#    - text_stats：统计文本字符数、词数
# 3）大模型自动决定是否调用工具、调用哪一个。
# 4）每轮都会在终端打印：模型的工具调用 & 工具执行结果。
#
# 退出方式：在输入中敲 exit / quit / 退出（小写即可）。


from typing import Dict, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)


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
    # 英文：按空格粗略分词；中文：因为没有空格，这里只是示意
    word_count = len(text.split())
    return f"这段文本共有 {char_count} 个字符，约 {word_count} 个词。"


def build_llm_with_tools():
    """
    构建一个带工具能力的 LLM：
    - 大脑：OpenAI Chat 模型（gpt-4.1-mini，可按你账号情况调整）
    - 工具：calculator + text_stats
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,  # 对话模式稍微放松一点
    )

    tools = [calculator, text_stats]
    llm_with_tools = llm.bind_tools(tools)
    tool_map: Dict[str, any] = {t.name: t for t in tools}

    return llm_with_tools, tool_map


def run_chat_loop():
    """
    进入一个持续对话循环：
    - 会话记忆：用 messages 列表保存整段对话历史（System + Human + AI + Tool）
    - 每轮：
        1）用户输入一句话
        2）模型可能发起 tool_calls
        3）本地执行工具
        4）把结果回传给模型
        5）输出最终回答
    """

    llm_with_tools, tool_map = build_llm_with_tools()

    # 会话级“记忆”：这里就是最简单、也是最直观的一种 Memory 实现
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "你是一个会合理使用工具的中文智能体。"
                "当需要精确计算或统计文本信息时，请优先调用工具。"
                "你要尽量参考对话历史，保持上下文连贯，并用通俗清晰的中文回答。"
            )
        )
    ]

    print("=========== 智能体对话模式 ===========")
    print("已加载工具：calculator, text_stats")
    print("提示：输入 exit / quit / 退出 可结束对话。")
    print("====================================\n")

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("智能体：好的，本次对话结束。")
            break

        if not user_input:
            continue

        # 将本轮用户输入加入“记忆”
        messages.append(HumanMessage(content=user_input))

        # 本轮工具使用记录（仅用于打印和思路清晰）
        tool_history: List[Tuple[str, dict, str]] = []

        print("\n[调试] 发送给模型的消息数量：", len(messages))

        # 第一次调用：看看模型是否要用工具
        response = llm_with_tools.invoke(messages)

        print("\n[调试] 模型第一次回复（可能包含工具调用）:")
        print(response)
        print("----------------------------------------")

        # 先把模型这次回复加入历史
        messages.append(response)

        # 如果没有工具调用，直接输出内容
        if not getattr(response, "tool_calls", None):
            print("\n智能体：", response.content, "\n")
            continue

        # 有工具调用：逐个执行
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"\n[调试] 模型决定调用工具：{tool_name}，参数：{tool_args}")

            if tool_name not in tool_map:
                tool_output = f"错误：未找到工具 {tool_name}"
            else:
                tool_output = tool_map[tool_name].invoke(tool_args)

            print(f"[调试] 工具 {tool_name} 执行结果：{tool_output}")

            tool_history.append((tool_name, tool_args, str(tool_output)))

            # 把工具结果封装成 ToolMessage，加到消息历史中
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_id,
                )
            )

        # 第二次调用：基于工具结果给出最终答案
        final_response = llm_with_tools.invoke(messages)

        messages.append(final_response)

        # 输出给你看的部分：模型的自然语言回答 + 简要工具总结
        print("\n智能体：", final_response.content)

        if tool_history:
            print("\n（本轮工具使用情况：）")
            for name, args, output in tool_history:
                print(f"- 调用了 {name}，参数 {args}，输出：{output}")

        print("\n----------------------------------------\n")


def main():
    run_chat_loop()


if __name__ == "__main__":
    main()
