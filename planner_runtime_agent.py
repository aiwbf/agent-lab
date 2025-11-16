# planner_runtime_agent.py
# 带“任务规划器（Planner）”的智能体运行时：
# 流程：
#   用户输入 -> 规划器LLM生成分步计划 -> 执行器LLM + 工具按计划完成任务 -> 输出回答

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
#   带 Planner 的智能体运行时
# ------------------------------

class PlannerRuntime:

    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        初始化运行时：
        - planner_llm：只负责“任务拆解/规划”，不调用工具
        - exec_llm_with_tools：负责“按计划执行任务 + 调用工具”
        """

        # 规划器 LLM：偏理性、结构化
        self.planner_llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            timeout=20,
            max_retries=2,
        )

        # 执行器 LLM：绑定工具，负责实际完成任务
        exec_llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            timeout=20,
            max_retries=2,
        )

        self.tools = [calculator, text_stats]
        self.exec_llm_with_tools = exec_llm.bind_tools(self.tools)
        self.tool_map: Dict[str, Any] = {t.name: t for t in self.tools}

        # 会话记忆：统一放在一个 message history 里
        self.messages: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "你是一个专业中文智能体，会根据 Planner 给出的分步计划执行任务。"
                    "你可以使用工具，也可以直接回答。请始终用清晰、通俗的中文解释你的过程。"
                )
            )
        ]

    # ------------------------------------------
    #             任务规划器（Planner）
    # ------------------------------------------
    def plan_task(self, user_input: str) -> str:
        """
        使用专门的 planner_llm，对用户任务进行拆解，生成一个“分步计划”。
        注意：这里只做文本规划，不调用工具。
        """

        system_prompt = SystemMessage(
            content=(
                "你是一个任务规划器（Planner）。"
                "你的目标是把用户的任务拆分成 2-5 个清晰的步骤。"
                "每一步都用中文描述，可以包括是否需要使用计算器或文本统计工具。"
                "只输出步骤列表，不要直接给出最终答案。"
                "示例：\n"
                "步骤1：先估算 23 * 47 的大致范围。\n"
                "步骤2：使用计算器工具计算精确结果。\n"
                "步骤3：整理结果并用通俗中文解释。"
            )
        )

        user_msg = HumanMessage(content=user_input)

        print("[调试][Planner] 正在为本轮任务生成分步计划...")
        try:
            response = self.planner_llm.invoke([system_prompt, user_msg])
        except Exception as e:
            print("[错误][Planner] 调用规划器 LLM 失败：", e)
            return "步骤1：直接尝试使用可用工具或语言能力解决问题。\n步骤2：如果失败，再给出错误提示。"

        plan_text = str(response.content)
        print("[调试][Planner] 生成的计划如下：\n", plan_text, "\n")

        return plan_text

    # ------------------------------------------
    #          工具调用处理逻辑
    # ------------------------------------------
    def _process_tool_calls(self, response) -> List[Tuple[str, dict, str]]:
        """执行执行器 LLM 给出的 tool_calls，并返回执行记录"""

        tool_history = []
        if not getattr(response, "tool_calls", None):
            return tool_history

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[调试][Exec] 模型调用工具：{tool_name}, 参数：{tool_args}")

            if tool_name not in self.tool_map:
                tool_output = f"未找到工具：{tool_name}"
            else:
                tool_output = self.tool_map[tool_name].invoke(tool_args)

            print(f"[调试][Exec] 工具输出：{tool_output}")

            # 把工具结果返回给执行器 LLM
            self.messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            )

            tool_history.append((tool_name, tool_args, str(tool_output)))

        return tool_history

    # ------------------------------------------
    #          单轮对话（含规划）
    # ------------------------------------------
    def chat_once(self, user_input: str) -> Tuple[str, List, str]:
        """
        一轮对话包含：
        1）Planner 拆解任务，生成 plan_text
        2）把 plan_text 加入历史（作为“执行建议”的 System 提示）
        3）执行器 LLM + 工具按计划完成任务，给出最终回答
        """

        # 1）先加入用户输入
        self.messages.append(HumanMessage(content=user_input))

        # 2）调用 Planner 生成计划
        plan_text = self.plan_task(user_input)

        # 把规划结果加入 messages，作为“执行提示”
        self.messages.append(
            SystemMessage(
                content=(
                    "以下是你在执行用户任务时应当参考的分步计划：\n"
                    f"{plan_text}\n"
                    "请按照这些步骤，有条理地完成任务。"
                )
            )
        )

        # 3）第一次调用执行器 LLM，看是否要用工具
        print("[调试][Exec] 准备调用执行器 LLM（第一次）")
        try:
            response = self.exec_llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[错误][Exec] 第一次调用执行器 LLM 失败：", e)
            return "执行任务时调用大模型失败，可能是网络或限流问题。", [], plan_text

        print("[调试][Exec] 已收到执行器第一次回复")
        self.messages.append(response)
        print("[调试][Exec] 第一次回复：", response)

        # 4）执行工具调用
        tool_history = self._process_tool_calls(response)

        # 5）第二次调用执行器 LLM，基于工具结果给出最终答案
        print("[调试][Exec] 准备调用执行器 LLM（第二次）")
        try:
            final_response = self.exec_llm_with_tools.invoke(self.messages)
        except Exception as e:
            print("[错误][Exec] 第二次调用执行器 LLM 失败：", e)
            return "第二次调用执行器失败，可能是网络或限流问题。", tool_history, plan_text

        print("[调试][Exec] 已收到执行器第二次回复")
        self.messages.append(final_response)

        return str(final_response.content), tool_history, plan_text

    # ------------------------------------------
    #     持续对话 loop（REPL）
    # ------------------------------------------
    def chat_loop(self):
        print("====== 带 Planner 的专业智能体（PlannerRuntime）======")
        print("我会先为你的任务生成分步计划，然后再执行。")
        print("已加载工具：calculator, text_stats")
        print("输入 exit / quit / 退出 来结束对话。")
        print("====================================================\n")

        while True:
            user_input = input("你：").strip()

            if user_input.lower() in ["exit", "quit", "退出"]:
                print("智能体：好的，结束对话。")
                break

            if not user_input:
                continue

            answer, tool_history, plan_text = self.chat_once(user_input)

            print("\n[本轮任务规划（Planner 输出）]")
            print(plan_text)

            print("\n智能体最终回答：", answer)

            if tool_history:
                print("\n（本轮工具调用记录）")
                for name, args, out in tool_history:
                    print(f"- {name} 参数={args}, 输出={out}")

            print("\n----------------------------------------------------\n")


# ------------------------------
#              主程序
# ------------------------------

def main():
    runtime = PlannerRuntime()
    runtime.chat_loop()


if __name__ == "__main__":
    main()
