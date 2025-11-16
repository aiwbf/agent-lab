# agent.py
# 这是你的第一个“真正的智能体”：
# 使用 OpenAI 的工具调用功能 + LangChain 的工具封装
# 能够：
# 1）理解你的任务
# 2）判断是否需要用工具
# 3）调用计算器工具
# 4）根据工具结果给出最终中文解释

from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# 1. 定义一个工具：计算器
@tool
def calculator(expression: str) -> str:
    """一个简单的计算器工具，输入算式（例如：'23 * 47'），返回计算结果。"""
    try:
        result = eval(expression)
        return f"表达式 {expression} 的计算结果是 {result}"
    except Exception as e:
        return f"计算出错：{e}"


def build_llm_with_tools():
    """
    构建一个带工具能力的 LLM：
    - 大脑：OpenAI Chat 模型
    - 工具：calculator（后续可以扩展多个工具）
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",  # 可根据你账号支持情况调整
        temperature=0.2,
    )

    # 把工具绑定到 LLM 上，让大模型知道“我有哪些工具可以用”
    tools = [calculator]
    llm_with_tools = llm.bind_tools(tools)

    # 为了后面方便根据名字找到工具，实现一个 {工具名: 工具对象} 的字典
    tool_map: Dict[str, any] = {t.name: t for t in tools}

    return llm_with_tools, tool_map


def run_agent(task: str):
    """
    运行一个简单的“单轮 Agent”流程：
    1）把用户任务发给大模型（带工具信息）
    2）模型决定要不要调用工具
    3）如果调用工具：
        a. 在本地真正执行该工具
        b. 把工具结果作为 ToolMessage 返回给模型
        c. 让模型基于工具输出给出最终答案
    """

    # 1. 构建带工具能力的 LLM 和工具映射表
    llm_with_tools, tool_map = build_llm_with_tools()

    # 2. 构建对话消息列表（messages）
    messages = [HumanMessage(content=task)]

    print("====== 发送任务给智能体 ======")
    print(task)
    print("================================\n")

    # 3. 第一次调用模型：看它会不会发出“工具调用请求”
    response = llm_with_tools.invoke(messages)

    print("---- 模型第一次回复（可能包含工具调用） ----")
    print(response)
    print("----------------------------------------\n")

    # 4. 检查是否有工具调用（tool_calls）
    if not getattr(response, "tool_calls", None):
        # 没有工具调用，说明模型认为自己可以直接回答
        print("（模型没有调用任何工具，直接返回答案）\n")
        return response.content

    # 5. 如果有工具调用，逐个执行
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"模型决定调用工具：{tool_name}，参数：{tool_args}")

        if tool_name not in tool_map:
            tool_output = f"错误：未找到工具 {tool_name}"
        else:
            # 真正在本地执行工具
            tool_output = tool_map[tool_name].invoke(tool_args)

        print(f"工具 {tool_name} 执行结果：{tool_output}\n")

        # 把工具执行结果作为 ToolMessage 追加到对话消息中
        messages.append(response)  # 模型的带 tool_calls 回复
        messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
            )
        )

    # 6. 再次调用模型，让它根据工具结果给出最终回答
    final_response = llm_with_tools.invoke(messages)

    print("---- 模型基于工具结果给出的最终回复 ----")
    print(final_response)
    print("----------------------------------------\n")

    return final_response.content


def main():
    # 一个稍微复杂一点的任务：要求模型先估算，再精确计算，再解释过程
    user_task = (
        "请先心算估算一下 23 * 47 大概是多少范围，"
        "然后再使用你提供的计算器工具给出精确结果，"
        "最后用通俗中文解释你是怎么得到答案的。"
    )

    final_answer = run_agent(user_task)

    print("====== 智能体最终回答 ======")
    print(final_answer)
    print("================================")


if __name__ == "__main__":
    main()
