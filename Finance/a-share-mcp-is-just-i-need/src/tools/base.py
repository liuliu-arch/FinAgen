"""
MCP工具的基础工具模块
包含用于调用数据源的共享辅助函数
"""
import logging
from typing import Callable, Optional
import pandas as pd

from src.formatting.markdown_formatter import format_df_to_markdown
from src.data_source_interface import NoDataFoundError, LoginError, DataSourceError

logger = logging.getLogger(__name__)


def safe_data_source_call(
    tool_name: str,
    data_source_method: Callable,
    data_type_name: str,
    **kwargs
) -> str:
    """
    通用的安全数据源调用函数，统一处理所有异常和错误情况
    
    参数:
        tool_name: 工具名称，用于日志记录
        data_source_method: 要调用的数据源方法
        data_type_name: 数据类型名称（用于日志记录）
        **kwargs: 传递给数据源方法的关键字参数
        
    返回:
        Markdown格式的结果字符串或错误消息
    """
    try:
        # 调用数据源方法
        df = data_source_method(**kwargs)
        logger.info(f"Successfully retrieved {data_type_name} data.")
        return format_df_to_markdown(df)
        
    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(f"Unexpected Exception processing {tool_name}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def call_financial_data_tool(
    tool_name: str,
    # 传递绑定的方法，如 active_data_source.get_profit_data
    data_source_method: Callable,
    data_type_name: str,
    code: str,
    year: str,
    quarter: int
) -> str:
    """
    用于减少财务数据工具重复代码的辅助函数

    参数:
        tool_name: 工具名称，用于日志记录
        data_source_method: 要在数据源上调用的方法
        data_type_name: 财务数据类型（用于日志记录）
        code: 股票代码
        year: 查询年份
        quarter: 查询季度

    返回:
        包含结果的Markdown格式字符串或错误消息
    """
    logger.info(f"Tool '{tool_name}' called for {code}, {year}Q{quarter}")
    try:
        # 基本验证
        if not year.isdigit() or len(year) != 4:
            logger.warning(f"Invalid year format requested: {year}")
            return f"Error: Invalid year '{year}'. Please provide a 4-digit year."
        if not 1 <= quarter <= 4:
            logger.warning(f"Invalid quarter requested: {quarter}")
            return f"Error: Invalid quarter '{quarter}'. Must be between 1 and 4."

        # 调用已实例化的active_data_source上的相应方法
        df = data_source_method(code=code, year=year, quarter=quarter)
        logger.info(
            f"Successfully retrieved {data_type_name} data for {code}, {year}Q{quarter}.")
        # 对财务表格使用较小的限制？
        return format_df_to_markdown(df)

    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError for {code}, {year}Q{quarter}: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError for {code}: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError for {code}: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError processing request for {code}: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(
            f"Unexpected Exception processing {tool_name} for {code}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def call_macro_data_tool(
    tool_name: str,
    data_source_method: Callable,
    data_type_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs  # 用于额外参数，如 year_type
) -> str:
    """
    宏观经济数据工具的辅助函数

    参数:
        tool_name: 工具名称，用于日志记录
        data_source_method: 要在数据源上调用的方法
        data_type_name: 数据类型（用于日志记录）
        start_date: 可选的开始日期
        end_date: 可选的结束日期
        **kwargs: 传递给data_source_method的额外关键字参数

    返回:
        包含结果的Markdown格式字符串或错误消息
    """
    date_range_log = f"from {start_date or 'default'} to {end_date or 'default'}"
    kwargs_log = f", extra_args={kwargs}" if kwargs else ""
    logger.info(f"Tool '{tool_name}' called {date_range_log}{kwargs_log}")
    
    return safe_data_source_call(
        tool_name,
        data_source_method,
        data_type_name,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def call_index_constituent_tool(
    tool_name: str,
    data_source_method: Callable,
    index_name: str,
    date: Optional[str] = None
) -> str:
    """
    指数成分股工具的辅助函数

    参数:
        tool_name: 工具名称，用于日志记录
        data_source_method: 要在数据源上调用的方法
        index_name: 指数名称（用于日志记录）
        date: 可选的查询日期

    返回:
        包含结果的Markdown格式字符串或错误消息
    """
    log_msg = f"Tool '{tool_name}' called for date={date or 'latest'}"
    logger.info(log_msg)
    
    return safe_data_source_call(
        tool_name,
        data_source_method,
        index_name,
        date=date
    )
