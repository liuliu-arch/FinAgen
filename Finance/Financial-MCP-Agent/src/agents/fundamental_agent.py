"""
FundamentalAnalysis Agent: Performs fundamental analysis of a stock using ReAct Agent framework.
基本面分析 Agent：使用ReAct Agent框架对股票进行基本面分析
"""
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


async def fundamental_agent(state: AgentState) -> AgentState:
    """
    使用ReAct框架进行基本面分析，直接集成MCP工具
    
    Args:
        state: 包含用户查询的当前 Agent状态

    Returns:
        更新后的AgentState，包含基本面分析结果
    """
    logger.info(
        f"{WAIT_ICON} FundamentalAgent: Starting fundamental analysis using ReAct framework.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "fundamental_agent"

    # 从状态中提取当前数据、消息和元数据
    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    # 记录 Agent开始执行，包含关键信息
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    # 验证用户查询是否存在
    if not user_query:
        logger.error(
            f"{ERROR_ICON} FundamentalAgent: User query is missing in state data.")
        current_data["fundamental_analysis_error"] = "User query is missing."

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, 0, False, "User query is missing")

        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    try:
        # 使用API调用
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        # 验证必要的环境变量是否存在
        if not all([api_key, base_url, model_name]):
            logger.error(
                f"{ERROR_ICON} FundamentalAgent: Missing OpenAI environment variables.")
            current_data["fundamental_analysis_error"] = "Missing OpenAI environment variables."

            # 记录 Agent执行失败
            execution_logger.log_agent_complete(agent_name, current_data, time.time(
            ) - agent_start_time, False, "Missing OpenAI environment variables")

            return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

        logger.info(
            f"{WAIT_ICON} FundamentalAgent: Creating ChatOpenAI with model {model_name}")
        # 创建LLM实例，设置合适的参数
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,  # 较低的温度确保分析的一致性
            max_tokens=6000,  # 增加token数量用于详细分析
            extra_body={"thinking": {"type": "disabled"}},
        )

        # 2. 获取MCP工具集
        logger.info(f"{WAIT_ICON} FundamentalAgent: Fetching MCP tools...")
        try:
            mcp_tools = await get_mcp_tools()
            if not mcp_tools:
                logger.error(
                    f"{ERROR_ICON} FundamentalAgent: No MCP tools available.")
                current_data["fundamental_analysis_error"] = "No MCP tools available."

                # 记录 Agent执行失败
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "No MCP tools available")

                return {"data": current_data, "messages": current_messages, "metadata": current_metadata}

            logger.info(
                f"{SUCCESS_ICON} FundamentalAgent: Successfully loaded {len(mcp_tools)} tools.")

            # 打印可用工具列表，便于调试
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Available tools: {tool_names}")

            # 3. 创建ReAct Agent - 只传入LLM和工具
            logger.info(
                f"{WAIT_ICON} FundamentalAgent: Creating ReAct agent...")
            agent = create_react_agent(llm, mcp_tools)

            # 4. 准备输入数据，构建详细的分析请求
            stock_code = current_data.get('stock_code', 'Unknown')
            company_name = current_data.get('company_name', 'Unknown')
            current_time_info = current_data.get('current_time_info', '未知时间')
            current_date = current_data.get('current_date', '未知日期')

            # 构建详细的基本面分析请求，包含多个分析维度
            agent_input = f"""请分析{company_name}（股票代码：{stock_code}）的基本面情况。

当前时间：{current_time_info}
当前日期：{current_date}

请进行以下基本面分析：
1. 获取公司基本信息和行业背景
2. 获取最新财务报表数据（资产负债表、利润表、现金流量表）
3. 分析盈利能力指标（毛利率、净利率、ROE等）
4. 分析成长能力指标（收入增长率、利润增长率等）
5. 分析运营效率指标（应收周转率、存货周转率等）
6. 分析偿债能力指标（资产负债率、流动比率等）
7. 查询历史分红情况
8. 提供基本面综合评估和投资价值分析

重要限制：请专注于财务数据和基本面指标分析，不要使用crawl_news工具获取新闻信息。基本面分析应该基于财务报表、财务指标和公司基本面数据，而不是新闻事件。

请使用可用的工具获取实际数据进行分析，而不是基于假设。如果某些数据无法获取，请尝试使用不同的时间周期或其他工具组合，基于可用信息提供尽可能全面的分析。"""

            logger.info(f"Agent input: {agent_input}")

            # 5. 调用ReAct Agent - 使用正确的messages格式
            logger.info(
                f"{WAIT_ICON} FundamentalAgent: Calling ReAct agent...")
            start_time = time.time()

            # LangGraph ReAct Agent需要messages格式的输入
            input_data = {
                "messages": [HumanMessage(content=agent_input)]
            }

            # 调用 Agent执行分析
            response = await agent.ainvoke(input_data)

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(
                f"ReAct agent execution completed in {execution_time:.2f} seconds")

            # 6. 提取分析结果
            final_output = "No analysis generated."

            if "messages" in response and isinstance(response["messages"], list):
                messages = response["messages"]
                # 查找最后一条AI消息，这通常包含最终的分析结果
                ai_messages = [
                    msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    final_output = last_ai_message.content
                    logger.info(
                        f"Successfully extracted analysis from AI message.")
                else:
                    logger.warning("No AI messages found in response")
                    # 如果没有AI消息，尝试获取所有消息的内容
                    all_content = []
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            all_content.append(str(msg.content))
                    if all_content:
                        final_output = "\n".join(all_content)
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                logger.error(
                    f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

            logger.info(
                f"Final extracted analysis length: {len(final_output)} characters")
            print(f"FUNDAMENTALAGENT: {final_output}")
            # 7. 记录LLM交互，用于后续分析和优化
            model_config = {
                "model": model_name,
                "temperature": 0.3,
                "max_tokens": 6000,
                "api_base": base_url
            }
            
            execution_logger.log_llm_interaction(
                agent_name=agent_name,
                interaction_type="react_agent",
                input_messages=[{"role": "user", "content": agent_input}],
                output_content=final_output,
                model_config=model_config,
                execution_time=execution_time
            )

            logger.info(
                f"{SUCCESS_ICON} FundamentalAgent: Successfully completed fundamental analysis.")
            
            # 8. 更新状态，保存分析结果和元数据
            current_data["fundamental_analysis"] = final_output
            current_metadata["fundamental_agent_executed"] = True
            current_metadata["fundamental_agent_timestamp"] = str(time.time())
            current_metadata["fundamental_agent_execution_time"] = f"{execution_time:.2f} seconds"

            # 9. 添加消息记录，保持对话历史
            new_message = {"role": "assistant", "content": "基本面分析已完成"}
            updated_messages = current_messages + [new_message]

            # 记录 Agent执行成功
            total_execution_time = time.time() - agent_start_time
            execution_logger.log_agent_complete(agent_name, {
                "fundamental_analysis_length": len(final_output),
                "analysis_preview": final_output[:500] if len(final_output) > 500 else final_output,
                "llm_execution_time": execution_time,
                "total_execution_time": total_execution_time
            }, total_execution_time, True)

            return {
                "data": current_data,
                "messages": updated_messages,
                "metadata": current_metadata
            }

        except Exception as e:
            logger.error(
                f"{ERROR_ICON} FundamentalAgent: Error in MCP or agent execution: {e}", exc_info=True)
            current_data[
                "fundamental_analysis_error"] = f"Error in MCP or agent execution: {e}"
            current_data["fundamental_analysis"] = f"基本面分析过程中出现错误: {str(e)}"
            current_metadata["fundamental_agent_error"] = str(e)

            # 记录 Agent执行失败
            execution_logger.log_agent_complete(
                agent_name, current_data, time.time() - agent_start_time, False, str(e))

            return {
                "data": current_data,
                "messages": current_messages,
                "metadata": current_metadata
            }

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} FundamentalAgent: Error during execution: {e}", exc_info=True)
        current_data["fundamental_analysis_error"] = f"Error during execution: {e}"
        current_metadata["fundamental_agent_error"] = str(e)

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return {
            "data": current_data,
            "messages": current_messages,
            "metadata": current_metadata
        }


# 本地测试函数
async def test_fundamental_agent():
    """基本面分析 Agent的测试函数"""
    from src.utils.state_definition import AgentState
    from datetime import datetime

    # 准备测试数据，包含当前时间信息
    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                          "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    # 创建测试状态，模拟真实的用户查询
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际的财务状况",
            "stock_code": "sh.603871",
            "company_name": "嘉友国际",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await fundamental_agent(test_state)
    print("Fundamental Analysis Result:")
    print(result)
    print(result.get("data", {}).get("fundamental_analysis", "No analysis found"))

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fundamental_agent())
