"""AKShare adapter restricted to Sina-backed endpoints.

This avoids BaoStock and Eastmoney, which may be unreachable from some cloud
GPU nodes.  It intentionally returns an explanatory one-row DataFrame for
features that have no verified Sina equivalent, so an Agent can continue with
the available evidence instead of hanging on a network login.
"""

from datetime import datetime, timedelta
import json
import json
from typing import List, Optional

import akshare as ak
import pandas as pd

from .baostock_data_source import BaostockDataSource
from .data_source_interface import NoDataFoundError


class AkshareSinaDataSource(BaostockDataSource):
    """Drop-in replacement for the project's BaoStock implementation."""

    @staticmethod
    def _sina_code(code: str) -> str:
        value = code.lower().replace(".", "").strip()
        if value.startswith(("sh", "sz", "bj")):
            return value
        if value.startswith(("5", "6", "9")):
            return f"sh{value}"
        if value.startswith(("4", "8")):
            return f"bj{value}"
        return f"sz{value}"

    @staticmethod
    def _plain_code(code: str) -> str:
        return AkshareSinaDataSource._sina_code(code)[2:]

    @staticmethod
    def _target_report_date(year: str, quarter: int) -> str:
        endings = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
        if quarter not in endings:
            raise ValueError("quarter must be one of 1, 2, 3, 4")
        return f"{year}{endings[quarter]}"

    @staticmethod
    def _compact(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        fixed = [c for c in ("报告日", "类型", "币种", "更新日期") if c in df.columns]
        matched = [
            c for c in df.columns
            if c not in fixed and any(keyword in str(c) for keyword in keywords)
        ]
        # Avoid huge Markdown/tool responses while retaining the most useful data.
        columns = fixed + matched[:24]
        return df.loc[:, columns] if columns else df

    def _report(self, code: str, symbol: str, year: str, quarter: int) -> pd.DataFrame:
        df = ak.stock_financial_report_sina(
            stock=self._sina_code(code), symbol=symbol
        )
        if df.empty or "报告日" not in df.columns:
            raise NoDataFoundError(f"No {symbol} data for {code}")
        report_date = self._target_report_date(year, quarter)
        dates = df["报告日"].astype(str).str.replace("-", "", regex=False)
        result = df.loc[dates == report_date].copy()
        if result.empty:
            eligible = df.loc[dates <= report_date].copy()
            if eligible.empty:
                raise NoDataFoundError(f"No {symbol} data on or before {report_date}")
            result = eligible.head(1)
        return result

    def _two_period_reports(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        frames = []
        for report_type in ("利润表", "资产负债表", "现金流量表"):
            for report_year in (year, str(int(year) - 1)):
                try:
                    part = self._report(code, report_type, report_year, quarter)
                    part.insert(0, "报表", report_type)
                    frames.append(part)
                except NoDataFoundError:
                    continue
        if not frames:
            raise NoDataFoundError(f"No financial reports for {code}")
        return pd.concat(frames, ignore_index=True, sort=False)

    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if frequency not in {"d", "w", "m"}:
            raise ValueError("Sina adapter currently supports d/w/m frequencies")
        adjust = {"1": "qfq", "2": "hfq", "3": ""}.get(adjust_flag)
        if adjust is None:
            raise ValueError("adjust_flag must be 1, 2, or 3")
        df = ak.stock_zh_a_daily(
            symbol=self._sina_code(code),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust=adjust,
        )
        if df.empty:
            raise NoDataFoundError(f"No K-line data for {code}")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if frequency in {"w", "m"}:
            rule = "W-FRI" if frequency == "w" else "ME"
            aggregations = {
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "amount": "sum",
            }
            aggregations = {k: v for k, v in aggregations.items() if k in df.columns}
            df = df.set_index("date").resample(rule).agg(aggregations).dropna().reset_index()
        df.insert(1, "code", code)
        df["preclose"] = df["close"].shift(1)
        df["pctChg"] = (df["close"] / df["preclose"] - 1) * 100
        df["adjustflag"] = adjust_flag
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        if fields:
            existing = [field for field in fields if field in df.columns]
            return df.loc[:, existing]
        preferred = [
            "date", "code", "open", "high", "low", "close", "preclose",
            "volume", "amount", "turnover", "pctChg", "adjustflag",
        ]
        return df.loc[:, [c for c in preferred if c in df.columns]]

    def get_stock_basic_info(
        self, code: str, fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        sina_code = self._sina_code(code)
        spot = ak.stock_zh_a_spot()
        result = spot.loc[spot["代码"].astype(str) == sina_code].copy()
        if result.empty:
            raise NoDataFoundError(f"No stock information for {code}")
        result.insert(0, "code", code)
        result.insert(1, "code_name", result["名称"])
        result["tradeStatus"] = "1"
        if fields:
            existing = [field for field in fields if field in result.columns]
            return result.loc[:, existing]
        return result

    def get_trade_dates(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        end = pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d"))
        start = pd.Timestamp(start_date or (end - timedelta(days=45)))
        data = ak.stock_zh_index_daily(symbol="sh000001")
        dates = pd.to_datetime(data["date"])
        dates = dates[(dates >= start) & (dates <= end)]
        calendar = pd.date_range(start, end, freq="D")
        return pd.DataFrame({
            "calendar_date": calendar.strftime("%Y-%m-%d"),
            "is_trading_day": calendar.isin(dates).astype(int),
        })

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        df = ak.stock_zh_a_spot().copy()
        return df.rename(columns={"代码": "code", "名称": "code_name"})

    def crawl_news(self, query: str, top_k: int = 10) -> str:
        """Fetch structured company news without the legacy Baidu scraper.

        AKShare accepts either a stock code or a company keyword. Scoring is
        deliberately left to Financial-MCP-Agent's score_financial_news tool.
        """
        limit = max(1, min(int(top_k), 10))
        df = ak.stock_news_em(symbol=str(query).strip())
        if df.empty:
            return json.dumps(
                {"query": query, "count": 0, "news": []},
                ensure_ascii=False,
            )
        records = []
        for _, row in df.head(limit).iterrows():
            records.append({
                "title": str(row.get("新闻标题", "")),
                "content": str(row.get("新闻内容", "")),
                "published_at": str(row.get("发布时间", "")),
                "source": str(row.get("文章来源", "")),
                "url": str(row.get("新闻链接", "")),
            })
        return json.dumps(
            {"query": query, "count": len(records), "news": records},
            ensure_ascii=False,
        )

    def crawl_news(self, query: str, top_k: int = 10) -> str:
        """Fetch structured company news without the legacy Baidu scraper.

        AKShare accepts either a stock code or a company keyword. Scoring is
        deliberately left to Financial-MCP-Agent's score_financial_news tool.
        """
        limit = max(1, min(int(top_k), 10))
        df = ak.stock_news_em(symbol=str(query).strip())
        if df.empty:
            return json.dumps(
                {"query": query, "count": 0, "news": []},
                ensure_ascii=False,
            )
        records = []
        for _, row in df.head(limit).iterrows():
            records.append({
                "title": str(row.get("新闻标题", "")),
                "content": str(row.get("新闻内容", "")),
                "published_at": str(row.get("发布时间", "")),
                "source": str(row.get("文章来源", "")),
                "url": str(row.get("新闻链接", "")),
            })
        return json.dumps(
            {"query": query, "count": len(records), "news": records},
            ensure_ascii=False,
        )

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._report(code, "利润表", year, quarter)
        return self._compact(df, ["营业", "利润", "收益", "成本", "税"])

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._report(code, "资产负债表", year, quarter)
        return self._compact(
            df, ["资产", "负债", "所有者权益", "股东权益", "货币资金", "应收", "存货"]
        )

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._report(code, "现金流量表", year, quarter)
        return self._compact(df, ["现金", "经营活动", "投资活动", "筹资活动"])

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._two_period_reports(code, year, quarter)
        return self._compact(df, ["营业收入", "营业利润", "净利润", "资产总计", "所有者权益"])

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._two_period_reports(code, year, quarter)
        return self._compact(df, ["营业收入", "应收", "存货", "流动资产", "资产总计"])

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        df = self._two_period_reports(code, year, quarter)
        return self._compact(df, ["营业收入", "净利润", "资产总计", "所有者权益"])

    @staticmethod
    def _unavailable(feature: str) -> pd.DataFrame:
        return pd.DataFrame({
            "status": ["unavailable"],
            "message": [f"{feature} has no verified Sina endpoint on this server; continue with available tools."],
        })

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        return self._unavailable("dividend data")

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._unavailable("standalone adjustment factors")

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._unavailable("performance express reports")

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._unavailable("forecast reports")

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        return self._unavailable("industry classification")

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._unavailable("SZ50 constituents")

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._unavailable("HS300 constituents")

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._unavailable("ZZ500 constituents")

    def get_deposit_rate_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        return self._unavailable("deposit rates")

    def get_loan_rate_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        return self._unavailable("loan rates")

    def get_required_reserve_ratio_data(self, start_date=None, end_date=None, year_type="0") -> pd.DataFrame:
        return self._unavailable("required reserve ratios")

    def get_money_supply_data_month(self, start_date=None, end_date=None) -> pd.DataFrame:
        return self._unavailable("monthly money supply")

    def get_money_supply_data_year(self, start_date=None, end_date=None) -> pd.DataFrame:
        return self._unavailable("yearly money supply")
