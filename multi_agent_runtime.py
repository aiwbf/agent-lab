# multi_agent_runtime.py
# 多智能体协作版运行时：
#   - PlannerAgent：只负责“想清楚要怎么做”（任务拆解）
#   - WorkerAgent：负责“按步骤执行 + 调用工具 + 给出最终答复”
#
# 你输入问题 ->
#   PlannerAgent 生成分步计划 ->
#   WorkerAgent 参考计划 + 工具，完成任务并回答你

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
#            Tools
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
#        Planner Agent
# ------------------------------

class PlannerAgent:
    """只负责任务拆解，不调用工具"""

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            timeout=20,
            max_retries=2,
        )

    def plan(self, user_input: str, history_hint: str = "") -> str:
        """
        把用户任务拆成 2–5 步。
        history_hint 可以传入一些对话背景，但这里我们保持简单。
        """
        system_msg = SystemMessage(
            content=(
                "你是一个任务规划器（Planner）。"
                "你的目标是帮执行智能体把任务拆分成 2–5 个清晰的步骤。"
                "每一步用中文写清楚要做什么，可以提到是否需要使用计算器或文本统计工具。"
                "不要直接给出问题的最终数值答案，只给“怎么做”的步骤。"
            )
        )

        user_msg = HumanMessage(
            content=(
                f"用户的当前需求是：{user_input}\n"
                f"对话背景提示（可选）：{history_hint}\n"
                "请给出分步计划。"
            )
        )

        print("\n[Planner] 正在生成任务计划...")
        try:
            response = self.llm.invoke([system_msg, user_msg])
        except Exception as e:
            print("[Planner][错误] 调用规划器 LLM 失败：", e)
            return (
                "步骤1：直接尝试用已有工具或语言能力解决问题。\n"
                "步骤2：如果遇到无法完成的部分，向用户说明原因。"
            )

        plan_text = str(response.content)
        print("[Planner] 计划生成完毕：\n", plan_text, "\n")
        return plan_text


# ------------------------------
#        Worker Agent
# ------------------------------

class WorkerAgent:
    """负责按计划执行任务，调用工具，给出最终回答"""

    def __init__(self, model: str = "gpt-4.1-mini"):
        # 执行器 LLM，绑定工具
        exec_llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            timeout=20,
            max_retries=2,
        )

        self.tools = [calculator, text_stats]
        self.exec_llm_with_tools = exec_llm.bind_tools(self.tools)
        self.tool_map: Dict[str, Any] = {t.name: t for t in self.tools}

        # Worker 自己的对话记忆（和 Planner 分开）
        self.messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "你是一个执行智能体（Worker）。"
                    "你的任务是参考 Planner 给出的分步计划，运用你的工具，完成用户的任务。"
                    "你可以调用 calculator 和 text_stats 工具。"
                    "回答时，请用通俗中文，说明你大致按哪些步骤完成了任务。"
                )
            )
        ]

    def _process_tool_calls(self, response) -> List[Tuple[str, dict, str]]:
        """执行工具调用，并返回调用记录"""
        tool_history: List[Tuple[str, dict, str]] = []

        if not getattr(response, "tool_calls", None):
            return tool_history

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[Worker][调试] 模型调用工具：{tool_name}，参数：{tool_args}")

            if tool_name not in self.tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = self.tool_map[tool_name].invoke(tool_args)

            print(f"[Worker][调试] 工具输出：{tool_output}")

            # 追加工具消息
            self.messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

        return tool_history

    def execute(self, user_input: str, plan_text: str) -> Tuple[str, List[Tuple[str, dict, str]]]:
        """
        按 Planner 给出的 plan_text 来执行任务，并给出最终回答。
        """

        # 把用户输入 + 计划一起告诉 Worker
        self.messages.append(
            SystemMessage(
                content=(
                    "以下是 Planner 为当前任务生成的分步计划，请尽量按这些步骤执行：\n"
                    f"{plan_text}\n"
                )
            )
        )
        self.messages.append(HumanMessage(content=user_input))

        # 第一次调用：看看要不要用工具
        print("[Worker][调试] 准备调用执行器 LLM（第一次）")
        try:
            response = self.exec_llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[Worker][错误] 第一次调用执行器失败：", e)
            return "执行任务时调用大模型失败，可能是网络或限流问题。", []

        print("[Worker][调试] 已收到第一次回复")
        self.messages.append(response)
        print("[Worker][调试] 第一次回复：", response)

        # 工具执行
        tool_history = self._process_tool_calls(response)

        # 第二次调用：基于工具结果给出最终回答
        print("[Worker][调试] 准备调用执行器 LLM（第二次）")
        try:
            final_response = self.exec_llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[Worker][错误] 第二次调用执行器失败：", e)
            return "第二次调用执行器失败，可能是网络或限流问题。", tool_history

        print("[Worker][调试] 已收到第二次回复")
        self.messages.append(final_response)

        return str(final_response.content), tool_history


# ------------------------------
#      Orchestrator 运行时
# ------------------------------

class MultiAgentRuntime:
    """
    一个最小可用的“多智能体协作”运行时：
    - planner: 专职拆任务
    - worker:  专职按计划执行 + 工具调用
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.planner = PlannerAgent(model=model)
        self.worker = WorkerAgent(model=model)

    def chat_loop(self):
        print("====== 多智能体协作运行时（Planner + Worker）======")
        print("流程：用户问题 -> Planner拆步骤 -> Worker按步骤执行 + 调工具")
        print("已加载工具：calculator, text_stats")
        print("输入 exit / quit / 退出 结束对话。")
        print("=================================================\n")

        while True:
            user_input = input("你：").strip()
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("智能体：好的，结束对话。")
                break

            if not user_input:
                continue

            # 1）Planner 拆解任务
            plan_text = self.planner.plan(user_input)

            # 2）Worker 按计划执行
            answer, tool_history = self.worker.execute(user_input, plan_text)

            # 3）在终端输出：计划 + 最终回答 + 工具调用记录
            print("\n[本轮 Planner 生成的计划]")
            print(plan_text)

            print("\n[Worker 最终回答]")
            print(answer)

            if tool_history:
                print("\n（Worker 工具调用记录）")
                for name, args, out in tool_history:
                    print(f"- {name} 参数={args}, 输出={out}")

            print("\n-------------------------------------------------\n")


# ------------------------------
#              主程序
# ------------------------------

def main():
    runtime = MultiAgentRuntime()
    runtime.chat_loop()


if __name__ == "__main__":
    main()
