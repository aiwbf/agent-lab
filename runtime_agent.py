# runtime_agent.py
# 专业级智能体架构：AgentRuntime（你的第一个“智能体运行时”）
# 本版改进：
# - 为调用大模型设置了 timeout 和 max_retries，避免长时间卡住
# - 在每次调用前后增加调试打印，帮助判断卡在哪一步
# - 对大模型调用增加 try/except，出现网络/Key/限流问题会给出清晰提示

from typing import Dict, List, Tuple, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)


# ------------------------------
#        工具定义区（Tools）
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
#     智能体运行时（核心类）
# ------------------------------

class AgentRuntime:

    def __init__(self, model: str = "gpt-4.1-mini"):
        """初始化运行时：大脑、工具、记忆"""
        # 加上 timeout 和 max_retries，避免“看起来死机”
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            timeout=20,     # 最多等 20 秒
            max_retries=2,  # 最多重试 2 次
        )

        # 注册所有工具
        self.tools = [calculator, text_stats]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 用于执行工具
        self.tool_map: Dict[str, Any] = {t.name: t for t in self.tools}

        # 会话记忆（Message History）
        self.messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "你是一个专业中文智能体，会根据需要自动调用工具。"
                    "在适当的时候使用工具，并在最终回答中给出清晰解释。"
                )
            )
        ]

    # ------------------------------------------
    #          工具调用处理逻辑
    # ------------------------------------------
    def _process_tool_calls(self, response) -> List[Tuple[str, dict, str]]:
        """处理一个模型响应中的所有 tool_calls，执行工具并返回执行信息"""

        tool_history = []

        if not getattr(response, "tool_calls", None):
            return tool_history

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[调试] 模型调用工具：{tool_name}, 参数：{tool_args}")

            if tool_name not in self.tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = self.tool_map[tool_name].invoke(tool_args)

            print(f"[调试] 工具输出：{tool_output}")

            # 返回工具结果给模型
            self.messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

        return tool_history

    # ------------------------------------------
    #          单轮智能体对话
    # ------------------------------------------
    def chat_once(self, user_input: str) -> Tuple[str, List]:
        """处理一次用户输入，包括工具调用与最终回答"""

        # 1）加入用户消息
        self.messages.append(HumanMessage(content=user_input))

        # 2）第一次调用：看是否要用工具
        print("[调试] 准备调用 llm_with_tools.invoke(...)（第一次）")
        try:
            response = self.llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[错误] 第一次调用大模型接口失败：", e)
            return "调用大模型接口失败，可能是网络、API Key 或限流问题。", []

        print("[调试] 已收到模型第一次回复")
        self.messages.append(response)

        print("[调试] 模型第一次回复：", response)

        # 3）执行工具调用（如果有）
        tool_history = self._process_tool_calls(response)

        # 4）第二次调用：基于工具结果生成最终答案
        print("[调试] 准备调用 llm_with_tools.invoke(...)（第二次）")
        try:
            final_response = self.llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[错误] 第二次调用大模型接口失败：", e)
            return "第二次调用大模型接口失败，可能是网络、API Key 或限流问题。", tool_history

        print("[调试] 已收到模型第二次回复")

        # 保存到历史
        self.messages.append(final_response)

        return final_response.content, tool_history

    # ------------------------------------------
    #     持续对话 loop（REPL）
    # ------------------------------------------
    def chat_loop(self):
        print("====== 专业智能体（AgentRuntime）======")
        print("你可以与我持续对话。我会自动决定是否调用工具。")
        print("输入 exit / quit / 退出 来结束对话。")
        print("======================================\n")

        while True:
            user_input = input("你：").strip()

            if user_input.lower() in ["exit", "quit", "退出"]:
                print("智能体：好的，结束对话。")
                break

            if not user_input:
                continue

            answer, tool_history = self.chat_once(user_input)

            print("\n智能体：", answer)

            if tool_history:
                print("\n（本轮工具调用记录）")
                for name, args, out in tool_history:
                    print(f"- {name} 参数={args}, 输出={out}")

            print("\n--------------------------------------\n")


# ------------------------------
#              主程序
# ------------------------------

def main():
    runtime = AgentRuntime()
    runtime.chat_loop()


if __name__ == "__main__":
    main()
