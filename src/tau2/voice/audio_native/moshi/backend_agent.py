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
    ) -> tuple[str, list[dict[str, Any]]]:  # 🎯 修改返回类型为 Tuple
        """运行 GPT-4o 的工具调用与信息压缩循环，并在本地执行它们，最终返回文本事实和被执行过的工具记录。"""
        # 🎯 新增：用于收集本轮循环中实际执行过的工具调用列表
        executed_tool_calls = []

        # 1. 将传入的自定义 Tool 对象转换为 OpenAI 兼容的 schema 字典格式
        openai_tools = []
        for tool in tools:
            # 方式 A：如果 tau2 提供了官方封装好的 openai_schema 方法/属性
            if hasattr(tool, "openai_schema"):
                schema = tool.openai_schema
                # 如果是方法则调用，如果是属性则直接取值
                schema_dict = schema() if callable(schema) else schema
                openai_tools.append(schema_dict)
            else:
                # 方式 B：从真实属性字段获取
                tool_name = getattr(tool, "name", "")
                tool_desc = getattr(tool, "short_desc", "") or getattr(tool, "long_desc", "")
                tool_params = getattr(tool, "params", {})
                if tool_name:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_desc,
                            "parameters": tool_params
                        }
                    })
        

        # 2. 拼接系统任务守则和运行时的操作守则
        runtime_directive = (
            "\n\n[BACKEND AGENT OPERATIONAL DIRECTIVE]\n"
            "You are acting as the backend tool-execution and information-compacting engine in a duplex voice system.\n\n"
            "CRITICAL MANDATE: EXTRACT PARAMETERS EXCLUSIVELY FROM 'USER' MESSAGES (ZERO TRUST IN ASSISTANT):\n"
            "1. ZERO TRUST IN ASSISTANT: Messages with role 'assistant' are spoken by a real-time speech model that constantly hallucinates, repeats outdated/wrong IDs, and makes severe phonetic errors (e.g. inventing random numbers or names).\n"
            "2. EXCLUSIVE PARAMETER SOURCE: You MUST extract entity parameters (reservation IDs, user IDs, customer names, emails, cancellation reasons) SOLELY and EXCLUSIVELY from messages where role is 'user'.\n"
            "3. STRICT PROHIBITION: You are STRICTLY FORBIDDEN from using, copying, or inferring ANY reservation ID, user ID, or email mentioned by the 'assistant'. Treat all assistant statements as completely untrusted noise regarding user entities.\n"
            "4. IMMEDIATE RESERVATION LOOKUP: Whenever the user mentions ANY 6-character alphanumeric code (e.g. 'EHGLP3'), treat it as the reservation ID and immediately call 'get_reservation_details(reservation_id=...)' as your FIRST action!\n\n"
            "Operational Flow & Multi-Step Reasoning:\n"
            "1. Carefully analyze the dialogue history and adhere to the task policy in the system prompt.\n"
            "2. You have the permission and responsibility to invoke tools multiple times in sequence (e.g., query reservation details first, check policy conditions, then proactively execute cancellation/refund or other required actions to resolve the user's intent).\n"
            "3. DO NOT output conversational pleasantries, spoken greetings, or filler (e.g., no 'Here is what I found', no 'Hello', no polite closing).\n"
            "4. DO NOT wrap your output in <tool_response> or any XML/HTML tags. Output ONLY the raw factual text (tags are added automatically by the system).\n"
            "5. Compress and flatten the raw tool outputs into a highly factual, dense, semi-structured summary, keeping all critical numbers, names, IDs, and exact refund amounts.\n\n"
            
            "Example Final Output (PURE TEXT, NO TAGS):\n"
            "Reservation EHGLP3 canceled successfully. Full refund of $54.00 processed back to original payment method ending in 7393."
        )
        combined_system_prompt = system_prompt + runtime_directive

        # 3. 构造给 GPT-4o 的完整消息历史，首位是 combined_system_prompt
        messages = [{"role": "system", "content": combined_system_prompt}] + conversation_history

        logger.info("Starting GPT-4o tool execution loop...")
        
        # 限制最大调用次数，防止由于模型或网络故障陷入无限死循环
        max_iterations = 10
        iteration = 0

        # 🎯 监控点 1：打印 GPT-4o 本次被唤醒时看到的完整历史
        print(f"\n{'='*30} [GPT-4O WAKE UP] {'='*30}", flush=True)
        print(f"Conversation turns seen by GPT-4o: {len(conversation_history)}", flush=True)
        for msg in conversation_history[-3:]: # 打印最近 3 轮
            print(f"  [{msg['role']}]: {msg['content'][:100]}...", flush=True)
        print(f"{'='*80}\n", flush=True)

        while iteration < max_iterations:
            iteration += 1
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0.0
                )
            except Exception as api_err:
                logger.error(f"Failed calling GPT-4o API: {api_err}")
                return "<tool_response>Error: Failed to contact backend brain.</tool_response>", []

            message = response.choices[0].message
            
            # 检查是否有工具调用请求
            if message.tool_calls:
                # 同步塞入 messages 历史中
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

                    logger.info(f"Executing tool: {func_name} with args: {func_args}")

                    print(
                        f"\n🛠️🛠️🛠️  [TOOL INVOKED] Function: '{func_name}'\n"
                        f"👉 Arguments: {json.dumps(func_args, ensure_ascii=False)}\n",
                        flush=True
                    )
                    
                    # 🎯 新增：记录当前这个工具调用（用来生成评测轨迹所需的 MoshiToolCallEvent）
                    executed_tool_calls.append({
                        "call_id": tool_call.id,
                        "name": func_name,
                        "arguments": func_args
                    })
                    
                    # 在本地 tools 中匹配并直接运行它
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
                            
                            print(
                                f"✅✅✅  [DB OUTCOME] Tool '{func_name}' returned:\n"
                                f"📄 {outcome_str[:300]}...\n",
                                flush=True
                            )

                        except Exception as exec_err:
                            outcome_str = f"Execution Error: {exec_err}"
                            logger.error(f"Error executing tool {func_name}: {exec_err}")
                    else:
                        outcome_str = f"Error: Tool '{func_name}' not found."
                        logger.warning(outcome_str)

                    # 反馈执行结果给 GPT-4o 上下文
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": outcome_str
                    })
                
                # 重新请求以进入下一轮分析
                continue

            else:
                # 没有新的工具调用，GPT-4o 决定给出最终总结
                final_content = message.content or ""
                logger.info(f"GPT-4o finished tool loop. Final output: {final_content}")
                
                # 兜底校准
                if "<tool_response>" not in final_content:
                    final_content = f"<tool_response>{final_content.strip()}</tool_response>"

                print(
                    f"\n📦📦📦  [GPT-4O FINAL FACT SUMMARY]:\n"
                    f"{final_content}\n"
                    f"{'='*80}\n",
                    flush=True
                )
                
                # 🎯 返回最终的事实总结和这轮循环里真正执行过的工具调用记录
                return final_content, executed_tool_calls

        logger.warning("Reached max iterations in GPT-4o tool loop.")
        return "<tool_response>Error: Max tool loop iterations reached.</tool_response>", executed_tool_calls