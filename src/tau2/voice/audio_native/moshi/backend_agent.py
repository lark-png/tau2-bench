import os
import json
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class BackendAgent:
    """负责驱动后端 GPT-4o 模型进行链式工具调用、原始数据压缩和格式化包装的认知引擎。"""

    def __init__(self, api_key: Optional[str] = None):
        # 自动获取环境变量中的 OPENAI_API_KEY
        actual_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not actual_key:
            logger.warning("OPENAI_API_KEY is not set in environment variables.")
        self.openai_client = AsyncOpenAI(api_key=actual_key)

    async def run_tool_loop(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Any],
    ) -> str:
        """运行 GPT-4o 的工具调用与信息压缩循环，直接在本地执行所有链式工具调用。"""
        # 1. 将传入的自定义 Tool 对象转换为 OpenAI 兼容的 schema 字典格式
        openai_tools = []
        for tool in tools:
            tool_name = getattr(tool, "name", "")
            tool_desc = getattr(tool, "description", "")
            tool_params = getattr(tool, "parameters", {})
            if tool_name:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc,
                        "parameters": tool_params
                    }
                })

        # 2. 拼接系统任务守则和运行时的操作守则（Operational Directive）
        runtime_directive = (
            "\n\n[BACKEND AGENT OPERATIONAL DIRECTIVE]\n"
            "You are acting as the backend tool-execution and information-compacting engine.\n\n"
            "Operational Flow & Multi-Step Reasoning:\n"
            "1. Carefully analyze the dialogue history. To resolve the user's request, you are expected to perform multi-step, sequential reasoning.\n"
            "2. You have the permission and responsibility to invoke tools multiple times in sequence (e.g., query a user ID first, then use that ID to fetch reservations). Do not hesitate to perform as many sequential tool calls as necessary to gather all required information.\n"
            "3. Once you have executed all necessary tools and collected the final results, do NOT output conversational responses or spoken greetings (such as 'Here is what I found...').\n"
            "4. Instead, compress and flatten the raw tool outputs into a highly factual, dense, semi-structured summary. Ensure that NO critical information is lost (such as reservation IDs, card digits, flight numbers, names, and exact pricing/amounts).\n"
            "5. You MUST wrap this factual compact summary inside <tool_response> and </tool_response> tags as your final output.\n\n"
            "Example Final Output format (when all tool calls are finished):\n"
            "<tool_response>User aarav_ahmed_6699 profile: Silver member... Reservation M20IZO canceled. Refund of $490 processed back to card 5018.</tool_response>"
        )
        combined_system_prompt = system_prompt + runtime_directive

        # 3. 构造给 GPT-4o 的完整消息历史，首位是我们的 combined_system_prompt
        messages = [{"role": "system", "content": combined_system_prompt}] + conversation_history

        logger.info("Starting GPT-4o tool execution loop...")
        
        # 限制最大调用次数，防止陷入死循环
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            try:
                # 调用 OpenAI 接口，为了保证提取参数的稳定与精准，温和度（temperature）设为 0.0
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0.0
                )
            except Exception as api_err:
                logger.error(f"Failed calling GPT-4o API: {api_err}")
                return "<tool_response>Error: Failed to contact backend brain.</tool_response>"

            message = response.choices[0].message
            
            # 检查是否有工具调用请求
            if message.tool_calls:
                # 将 GPT-4o 发出的调用塞入 messages 历史中进行上下文同步
                messages.append(message)
                logger.info(f"GPT-4o requested {len(message.tool_calls)} tool calls at iteration {iteration}.")
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args_str = tool_call.function.arguments
                    
                    try:
                        func_args = json.loads(func_args_str)
                    except Exception as json_err:
                        func_args = {}
                        logger.warning(f"Failed to parse tool arguments JSON: {json_err}")

                    print(
                        f"\n\n🛠️🛠️🛠️  [TOOL TRIGGERED] GPT-4o requested tool: '{func_name}' "
                        f"\n👉 Parameters: {func_args_str} 🛠️🛠️🛠️\n\n", 
                        flush=True
                    )

                    logger.info(f"Executing tool: {func_name} with args: {func_args}")
                    
                    # 在本地 tools 中匹配对应的 executable tool 对象并直接调用
                    target_tool = next((t for t in tools if getattr(t, "name", "") == func_name), None)
                    
                    if target_tool is not None:
                        try:
                            # 执行工具，兼容 invoke 或直接调用的协议
                            if hasattr(target_tool, "invoke"):
                                raw_outcome = target_tool.invoke(**func_args)
                            else:
                                raw_outcome = target_tool(**func_args)
                            
                            outcome_str = str(raw_outcome)
                            logger.info(f"Tool {func_name} executed successfully. Result length: {len(outcome_str)}")
                            short_outcome = outcome_str[:200] + ("..." if len(outcome_str) > 200 else "")
                            print(
                                f"\n\n✅✅✅  [TOOL EXECUTION SUCCESS] Tool: '{func_name}' "
                                f"\n📝 Outcome Summary: {short_outcome} ✅✅✅\n\n", 
                                flush=True
                            )
                        except Exception as exec_err:
                            outcome_str = f"Execution Error: {exec_err}"
                            logger.error(f"Error executing tool {func_name}: {exec_err}")
                            print(
                                f"\n\n❌❌❌  [TOOL EXECUTION FAILED] Tool: '{func_name}' "
                                f"\n⚠️ Error Message: {exec_err} ❌❌❌\n\n", 
                                flush=True
                            )
                    else:
                        outcome_str = f"Error: Tool '{func_name}' not found."
                        logger.warning(outcome_str)

                    # 将执行结果作为 tool 角色反馈给 GPT-4o 上下文
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": outcome_str
                    })
                
                # 继续下一次循环以让 GPT-4o 进行下一步评估
                continue

            else:
                # 没有工具调用了，说明 GPT-4o 已经生成了最终的事实压缩文本
                final_content = message.content or ""
                logger.info(f"GPT-4o finished tool loop. Final output: {final_content}")
                
                # 兜底校准
                if "<tool_response>" not in final_content:
                    final_content = f"<tool_response>{final_content.strip()}</tool_response>"
                print(f"GPT-4o finished tool loop. Final output: {final_content}")
                return final_content

        logger.warning("Reached max iterations in GPT-4o tool loop.")
        return "<tool_response>Error: Max tool loop iterations reached.</tool_response>"