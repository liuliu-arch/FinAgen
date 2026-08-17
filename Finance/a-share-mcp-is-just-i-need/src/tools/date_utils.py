"""
日期工具，用于MCP服务器
包含获取当前日期和最新交易日的工具
"""
import logging
from datetime import datetime, timedelta
import calendar

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)


def register_date_utils_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册日期工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    # @app.tool()
    # def get_current_date() -> str:
    #     """
    #     获取当前日期，可用于查询最新数据。

    #     Returns:
    #         当前日期，格式为'YYYY-MM-DD'。
    #     """
    #     logger.info("Tool 'get_current_date' called")
    #     current_date = datetime.now().strftime("%Y-%m-%d")
    #     logger.info(f"Returning current date: {current_date}")
    #     return current_date

    @app.tool()
    def get_latest_trading_date() -> str:
        """
        获取最近的交易日期。如果当天是交易日，则返回当天日期；否则返回最近的交易日。

        Returns:
            最近的交易日期，格式为'YYYY-MM-DD'。
        """
        logger.info("Tool 'get_latest_trading_date' called")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            # 获取当前日期前后一周的交易日历
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now().replace(day=28)).strftime("%Y-%m-%d")

            df = active_data_source.get_trade_dates(
                start_date=start_date, end_date=end_date)

            # 筛选出最近的交易日
            valid_trading_days = df[df['is_trading_day']
                                    == '1']['calendar_date'].tolist()

            # 找出小于等于今天的最大日期
            latest_trading_date = None
            for date in valid_trading_days:
                if date <= today and (latest_trading_date is None or date > latest_trading_date):
                    latest_trading_date = date

            if latest_trading_date:
                logger.info(
                    f"Latest trading date found: {latest_trading_date}")
                return latest_trading_date
            else:
                logger.warning(
                    "No trading dates found before today, returning today's date")
                return today

        except Exception as e:
            logger.exception(f"Error determining latest trading date: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    @app.tool()
    def get_market_analysis_timeframe(period: str = "recent") -> str:
        """
        获取适合市场分析的时间范围，基于当前真实日期而不是训练数据。
        这个工具应该在进行市场分析或大盘分析时首先调用，以确保使用最新的实际数据。

        参数:
            period: 时间范围类型，可选值:
                   "recent": 最近1-2个月(默认)
                   "quarter": 最近一个季度
                   "half_year": 最近半年
                   "year": 最近一年

        返回:
            包含分析时间范围的详细描述字符串，格式为"YYYY年M月-YYYY年M月"。
        """
        logger.info(
            f"Tool 'get_market_analysis_timeframe' called with period={period}")

        now = datetime.now()
        end_date = now

        # 根据请求的时间段确定开始日期
        if period == "recent":
            # 最近1-2个月
            if now.day < 15:
                # 如果当前是月初，看前两个月
                if now.month == 1:
                    start_date = datetime(now.year - 1, 11, 1)  # 前年11月
                    middle_date = datetime(now.year - 1, 12, 1)  # 前年12月
                elif now.month == 2:
                    start_date = datetime(now.year, 1, 1)  # 今年1月
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 2, 1)  # 两个月前
                    middle_date = datetime(now.year, now.month - 1, 1)  # 上个月
            else:
                # 如果当前是月中或月末，看前一个月到现在
                if now.month == 1:
                    start_date = datetime(now.year - 1, 12, 1)  # 前年12月
                    middle_date = start_date
                else:
                    start_date = datetime(now.year, now.month - 1, 1)  # 上个月
                    middle_date = start_date

        elif period == "quarter":
            # 最近一个季度 (约3个月)
            if now.month <= 3:
                start_date = datetime(now.year - 1, now.month + 9, 1)
            else:
                start_date = datetime(now.year, now.month - 3, 1)
            middle_date = start_date

        elif period == "half_year":
            # 最近半年
            if now.month <= 6:
                start_date = datetime(now.year - 1, now.month + 6, 1)
            else:
                start_date = datetime(now.year, now.month - 6, 1)
            middle_date = datetime(start_date.year, start_date.month + 3, 1) if start_date.month <= 9 else \
                datetime(start_date.year + 1, start_date.month - 9, 1)

        elif period == "year":
            # 最近一年
            start_date = datetime(now.year - 1, now.month, 1)
            middle_date = datetime(start_date.year, start_date.month + 6, 1) if start_date.month <= 6 else \
                datetime(start_date.year + 1, start_date.month - 6, 1)
        else:
            # 默认为最近1个月
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
            middle_date = start_date

        # 格式化为用户友好的显示
        def get_month_end_day(year, month):
            return calendar.monthrange(year, month)[1]

        # 确保结束日期不超过当前日期
        end_day = min(get_month_end_day(
            end_date.year, end_date.month), end_date.day)
        end_display_date = f"{end_date.year}年{end_date.month}月"
        end_iso_date = f"{end_date.year}-{end_date.month:02d}-{end_day:02d}"

        # 开始日期显示
        start_display_date = f"{start_date.year}年{start_date.month}月"
        start_iso_date = f"{start_date.year}-{start_date.month:02d}-01"

        # 如果跨年或时间段较长，添加年份显示
        if start_date.year != end_date.year:
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.year}年{end_date.month}月"
        elif middle_date.month != start_date.month and middle_date.month != end_date.month:
            # 如果是季度或半年，显示中间月份
            date_range = f"{start_date.year}年{start_date.month}月-{middle_date.month}月-{end_date.month}月"
        elif start_date.month != end_date.month:
            date_range = f"{start_date.year}年{start_date.month}月-{end_date.month}月"
        else:
            date_range = f"{start_date.year}年{start_date.month}月"

        result = f"{date_range} (ISO日期范围: {start_iso_date} 至 {end_iso_date})"
        logger.info(f"Generated market analysis timeframe: {result}")
        return result
