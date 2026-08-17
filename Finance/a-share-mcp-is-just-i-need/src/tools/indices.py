"""
指数相关工具，用于MCP服务器
包含获取指数成分股的工具
"""
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.tools.base import call_index_constituent_tool

logger = logging.getLogger(__name__)


def register_index_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册指数相关工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @app.tool()
    def get_stock_industry(code: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        获取指定股票或指定日期所有股票的行业分类数据

        参数:
            code: 可选的股票代码（例如：'sh.600000'）。如果为None，则获取所有股票的数据
            date: 可选的日期，格式为'YYYY-MM-DD'。如果为None，则使用最新可用日期

        返回:
            包含行业分类数据的Markdown表格或错误消息
        """
        log_msg = f"Tool 'get_stock_industry' called for code={code or 'all'}, date={date or 'latest'}"
        logger.info(log_msg)
        try:
            # 如果需要，可以添加日期验证
            df = active_data_source.get_stock_industry(code=code, date=date)
            logger.info(
                f"Successfully retrieved industry data for {code or 'all'}, {date or 'latest'}.")
            from src.formatting.markdown_formatter import format_df_to_markdown
            return format_df_to_markdown(df)

        except Exception as e:
            logger.exception(
                f"Exception processing get_stock_industry: {e}")
            return f"Error: An unexpected error occurred: {e}"

    @app.tool()
    def get_sz50_stocks(date: Optional[str] = None) -> str:
        """
        获取指定日期的深证50指数成分股数据

        参数:
            date: 可选的日期，格式为'YYYY-MM-DD'。如果为None，则使用最新可用日期

        返回:
            包含深证50指数成分股的Markdown表格或错误消息
        """
        return call_index_constituent_tool(
            "get_sz50_stocks",
            active_data_source.get_sz50_stocks,
            "深证50",
            date
        )

    @app.tool()
    def get_hs300_stocks(date: Optional[str] = None) -> str:
        """
        获取指定日期的沪深300指数成分股数据

        参数:
            date: 可选的日期，格式为'YYYY-MM-DD'。如果为None，则使用最新可用日期

        返回:
            包含沪深300指数成分股的Markdown表格或错误消息
        """
        return call_index_constituent_tool(
            "get_hs300_stocks",
            active_data_source.get_hs300_stocks,
            "沪深300",
            date
        )

    @app.tool()
    def get_zz500_stocks(date: Optional[str] = None) -> str:
        """
        获取指定日期的中证500指数成分股数据

        参数:
            date: 可选的日期，格式为'YYYY-MM-DD'。如果为None，则使用最新可用日期

        返回:
            包含中证500指数成分股的Markdown表格或错误消息
        """
        return call_index_constituent_tool(
            "get_zz500_stocks",
            active_data_source.get_zz500_stocks,
            "中证500",
            date
        )
