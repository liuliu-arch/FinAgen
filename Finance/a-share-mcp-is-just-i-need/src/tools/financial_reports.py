"""
财务报表相关工具，用于MCP服务器
"""
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource
from src.tools.base import call_financial_data_tool

logger = logging.getLogger(__name__)


def safe_financial_report_fetch(
    func_name: str,
    data_source_func,
    report_type: str,
    code: str,
    start_date: str = None,
    end_date: str = None,
    year: str = None,
    quarter: int = None
) -> str:
    """
    安全的财务报表数据获取函数，统一处理所有异常和错误情况
    
    参数:
        func_name: 函数名称，用于日志记录
        data_source_func: 数据源函数
        report_type: 报告类型描述
        code: 股票代码
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        year: 年份（可选）
        quarter: 季度（可选）
        
    返回:
        Markdown格式的数据表格或错误消息
    """
    try:
        # 根据参数类型调用不同的数据源函数
        if year and quarter:
            df = data_source_func(code=code, year=year, quarter=quarter)
        elif start_date and end_date:
            df = data_source_func(code=code, start_date=start_date, end_date=end_date)
        else:
            raise ValueError("Invalid parameters provided")
        
        logger.info(f"Successfully retrieved {report_type} data for {code}")
        from src.formatting.markdown_formatter import format_df_to_markdown
        return format_df_to_markdown(df)
        
    except Exception as e:
        logger.exception(f"Exception processing {func_name} for {code}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def register_financial_report_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册财务报表相关工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @app.tool()
    def get_profit_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度盈利能力数据（如ROE、净利润率等）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含盈利能力数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_profit_data",
            active_data_source.get_profit_data,
            "盈利能力",
            code, year, quarter
        )

    @app.tool()
    def get_operation_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度营运能力数据（如周转率等）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含营运能力数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_operation_data",
            active_data_source.get_operation_data,
            "营运能力",
            code, year, quarter
        )

    @app.tool()
    def get_growth_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度成长能力数据（如同比增长率等）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含成长能力数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_growth_data",
            active_data_source.get_growth_data,
            "成长能力",
            code, year, quarter
        )

    @app.tool()
    def get_balance_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度资产负债表/偿债能力数据（如流动比率、资产负债率等）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含资产负债表数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_balance_data",
            active_data_source.get_balance_data,
            "资产负债表",
            code, year, quarter
        )

    @app.tool()
    def get_cash_flow_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度现金流量数据（如CFO/营业收入比率等）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含现金流量数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_cash_flow_data",
            active_data_source.get_cash_flow_data,
            "现金流量",
            code, year, quarter
        )

    @app.tool()
    def get_dupont_data(code: str, year: str, quarter: int) -> str:
        """
        获取股票的季度杜邦分析数据（ROE分解）

        参数:
            code: 股票代码（例如：'sh.600000'）
            year: 4位数字年份（例如：'2023'）
            quarter: 季度（1、2、3或4）

        返回:
            包含杜邦分析数据的Markdown表格或错误消息
        """
        return call_financial_data_tool(
            "get_dupont_data",
            active_data_source.get_dupont_data,
            "杜邦分析",
            code, year, quarter
        )

    @app.tool()
    def get_performance_express_report(code: str, start_date: str, end_date: str) -> str:
        """
        获取股票在指定日期范围内的业绩快报数据
        注意：公司仅在特定情况下才需要发布这些报告

        参数:
            code: 股票代码（例如：'sh.600000'）
            start_date: 开始日期（报告发布日期），格式为'YYYY-MM-DD'
            end_date: 结束日期（报告发布日期），格式为'YYYY-MM-DD'

        返回:
            包含业绩快报数据的Markdown表格或错误消息
        """
        return safe_financial_report_fetch(
            "get_performance_express_report",
            active_data_source.get_performance_express_report,
            "业绩快报",
            code,
            start_date=start_date,
            end_date=end_date
        )

    @app.tool()
    def get_forecast_report(code: str, start_date: str, end_date: str) -> str:
        """
        获取股票在指定日期范围内的业绩预告数据
        注意：公司仅在特定情况下才需要发布这些报告

        参数:
            code: 股票代码（例如：'sh.600000'）
            start_date: 开始日期（报告发布日期），格式为'YYYY-MM-DD'
            end_date: 结束日期（报告发布日期），格式为'YYYY-MM-DD'

        返回:
            包含业绩预告数据的Markdown表格或错误消息
        """
        return safe_financial_report_fetch(
            "get_forecast_report",
            active_data_source.get_forecast_report,
            "业绩预告",
            code,
            start_date=start_date,
            end_date=end_date
        )

