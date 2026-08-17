# 使用Baostock库实现FinancialDataSource接口的具体数据源
import baostock as bs  # Baostock数据API库，用于获取A股市场数据
import pandas as pd    # 数据处理和分析库，用于处理返回的数据框
from typing import List, Optional, Dict  # 类型注解支持，增强代码可读性和类型检查
import logging         # 日志记录模块，用于跟踪程序执行和调试
from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError
from .utils import (
    baostock_login_context,  # 登录上下文管理器，自动处理登录登出
    fetch_financial_data,    # 通用财务数据获取函数
    fetch_index_constituent_data,  # 通用指数成分股数据获取函数
    fetch_macro_data,        # 通用宏观经济数据获取函数
    fetch_generic_data,      # 通用数据获取函数
    format_fields            # 字段格式化函数
)
import requests
from bs4 import BeautifulSoup
# 为当前模块创建专用的日志记录器，便于调试和错误追踪
logger = logging.getLogger(__name__)

# K线数据的默认字段，包含股票的基本交易信息和财务指标
DEFAULT_K_FIELDS = [
    "date",        # 交易日期
    "code",        # 股票代码
    "open",        # 开盘价
    "high",        # 最高价
    "low",         # 最低价
    "close",       # 收盘价
    "preclose",    # 前收盘价
    "volume",      # 成交量
    "amount",      # 成交金额
    "adjustflag",  # 复权类型标识
    "turn",        # 换手率
    "tradestatus", # 交易状态
    "pctChg",      # 涨跌幅
    "peTTM",       # 市盈率TTM
    "pbMRQ",       # 市净率MRQ
    "psTTM",       # 市销率TTM
    "pcfNcfTTM",   # 市现率TTM
    "isST"         # 是否ST股票
]

# 股票基本信息的默认字段
DEFAULT_BASIC_FIELDS = [
    "code",        # 股票代码
    "tradeStatus", # 交易状态
    "code_name"    # 股票名称
    # 可根据需要添加更多默认字段，如"industry"(行业), "listingDate"(上市日期)
]


class BaostockDataSource(FinancialDataSource):
    """
    使用Baostock library实现FinancialDataSource接口的实现类
    """

    def _format_fields(self, fields: Optional[List[str]], default_fields: List[str]) -> str:
        """
        将字段列表格式化为Baostock API所需的逗号分隔字符串
        
        参数:
            fields: 用户请求的字段列表（可选）
            default_fields: 默认字段列表（当fields为空时使用）
            
        返回:
            逗号分隔的字段字符串
            
        异常:
            ValueError: 如果请求的字段包含非字符串类型
        """
        return format_fields(fields, default_fields)

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度盈利能力数据"""
        return fetch_financial_data(bs.query_profit_data, "Profitability", code, year, quarter)

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度运营能力数据"""
        return fetch_financial_data(bs.query_operation_data, "Operation Capability", code, year, quarter)

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度成长能力数据"""
        return fetch_financial_data(bs.query_growth_data, "Growth Capability", code, year, quarter)

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度资产负债表数据（偿债能力）"""
        return fetch_financial_data(bs.query_balance_data, "Balance Sheet", code, year, quarter)

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度现金流量数据"""
        return fetch_financial_data(bs.query_cash_flow_data, "Cash Flow", code, year, quarter)

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度杜邦分析数据"""
        return fetch_financial_data(bs.query_dupont_data, "DuPont Analysis", code, year, quarter)

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取深证50指数成分股"""
        return fetch_index_constituent_data(bs.query_sz50_stocks, "SZSE 50", date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取沪深300指数成分股"""
        return fetch_index_constituent_data(bs.query_hs300_stocks, "CSI 300", date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取中证500指数成分股"""
        return fetch_index_constituent_data(bs.query_zz500_stocks, "CSI 500", date)

    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取基准存款利率"""
        return fetch_macro_data(bs.query_deposit_rate_data, "Deposit Rate", start_date, end_date)

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取基准贷款利率"""
        return fetch_macro_data(bs.query_loan_rate_data, "Loan Rate", start_date, end_date)

    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        """使用Baostock获取存款准备金率数据"""
        # 注意额外的yearType参数通过kwargs处理
        return fetch_macro_data(bs.query_required_reserve_ratio_data, "Required Reserve Ratio", start_date, end_date, yearType=year_type)

    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取月度货币供应量数据（M0、M1、M2）"""
        # Baostock期望这里的日期格式为YYYY-MM
        return fetch_macro_data(bs.query_money_supply_data_month, "Monthly Money Supply", start_date, end_date)

    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取年度货币供应量数据（M0、M1、M2 - 年末余额）"""
        # Baostock期望这里的日期格式为YYYY
        return fetch_macro_data(bs.query_money_supply_data_year, "Yearly Money Supply", start_date, end_date)

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指定时间范围内的交易日历数据"""
        return fetch_macro_data(bs.query_trade_dates, "Trade Dates", start_date, end_date)

    def get_historical_k_data(
        self,
        code: str,                    # 股票代码（如"sh.600000"）
        start_date: str,              # 开始日期（如"2023-01-01"）
        end_date: str,                # 结束日期（如"2023-12-31"）
        frequency: str = "d",         # 数据频率：d=日线，w=周线，m=月线
        adjust_flag: str = "3",       # 复权类型：1=前复权，2=后复权，3=不复权
        fields: Optional[List[str]] = None,  # 可选字段列表
    ) -> pd.DataFrame:
        """获取股票历史K线数据"""
        logger.info(
            f"Fetching K-data for {code} ({start_date} to {end_date}), freq={frequency}, adjust={adjust_flag}")
        
        try:
            # 格式化请求字段，如果未指定则使用默认K线字段
            formatted_fields = self._format_fields(fields, DEFAULT_K_FIELDS)
            logger.debug(
                f"Requesting fields from Baostock: {formatted_fields}")

            # 使用登录上下文管理器确保API连接
            with baostock_login_context():
                # 调用Baostock API获取K线数据
                rs = bs.query_history_k_data_plus(
                    code,
                    formatted_fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjust_flag
                )

                # 检查API返回的错误码
                if rs.error_code != '0':
                    logger.error(
                        f"Baostock API error (K-data) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    
                    # 区分"无数据"和"API错误"两种情况
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(
                            f"No historical data found for {code} in the specified range. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching K-data: {rs.error_msg} (code: {rs.error_code})")

                # 遍历结果集，收集所有数据行
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                # 检查是否为空结果集
                if not data_list:
                    logger.warning(
                        f"No historical data found for {code} in range (empty result set from Baostock).")
                    raise NoDataFoundError(
                        f"No historical data found for {code} in the specified range (empty result set).")

                # 将数据转换为DataFrame，使用API返回的字段名作为列名
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # 已知异常直接重新抛出
            logger.warning(
                f"Caught known error fetching K-data for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            # 未知异常包装为DataSourceError
            logger.exception(
                f"Unexpected error fetching K-data for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching K-data for {code}: {e}")

    def get_stock_basic_info(self, code: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取股票基本信息（如股票名称、交易状态等）"""
        logger.info(f"Fetching basic info for {code}")
        
        try:
            # 记录调试信息：请求的字段
            logger.debug(
                f"Requesting basic info for {code}. Optional fields requested: {fields}")

            # 使用登录上下文管理器
            with baostock_login_context():
                # 调用Baostock API获取股票基本信息
                rs = bs.query_stock_basic(code=code)

                # 检查API错误
                if rs.error_code != '0':
                    logger.error(
                        f"Baostock API error (Basic Info) for {code}: {rs.error_msg} (code: {rs.error_code})")
                    
                    # 区分无数据和API错误
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(
                            f"No basic info found for {code}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching basic info: {rs.error_msg} (code: {rs.error_code})")

                # 收集数据行
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                # 检查空结果
                if not data_list:
                    logger.warning(
                        f"No basic info found for {code} (empty result set from Baostock).")
                    raise NoDataFoundError(
                        f"No basic info found for {code} (empty result set).")

                # 转换为DataFrame
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved basic info for {code}. Columns: {result_df.columns.tolist()}")

                # 如果用户指定了字段，则筛选返回的列
                if fields:
                    # 找出用户请求的字段中实际存在的列
                    available_cols = [
                        col for col in fields if col in result_df.columns]
                    
                    # 如果用户请求的字段都不存在，则报错
                    if not available_cols:
                        raise ValueError(
                            f"None of the requested fields {fields} are available in the basic info result.")
                    
                    logger.debug(
                        f"Selecting columns: {available_cols} from basic info for {code}")
                    result_df = result_df[available_cols]

                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # 已知异常重新抛出
            logger.warning(
                f"Caught known error fetching basic info for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            # 未知异常包装
            logger.exception(
                f"Unexpected error fetching basic info for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching basic info for {code}: {e}")

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        """获取股票分红派息数据"""
        return fetch_generic_data(
            bs.query_dividend_data,
            "Dividend",
            code=code,
            year=year,
            yearType=year_type
        )

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票复权因子数据"""
        return fetch_generic_data(
            bs.query_adjust_factor,
            "Adjustment Factor",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票业绩快报数据（业绩快报）"""
        return fetch_generic_data(
            bs.query_performance_express_report,
            "Performance Express Report",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票业绩预告数据（业绩预告）"""
        return fetch_generic_data(
            bs.query_forecast_report,
            "Performance Forecast Report",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        """获取股票行业分类数据"""
        return fetch_generic_data(
            bs.query_stock_industry,
            "Industry",
            code=code,
            date=date
        )

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取指定日期的全市场股票列表"""
        return fetch_generic_data(
            bs.query_all_stock,
            "All Stock List",
            day=date
        )
    # 新增爬虫功能
    def crawl_news(self, query: str, top_k: int = 10) -> str:
        """
        直接从浏览器搜索并爬取相关文章内容，并使用风险模型和情感模型进行分析
        
        Args:
            query: 用户查询
            top_k: 返回的新闻数量
            
        Returns:
            格式化的新闻结果
        """
        try:
            
            
            # 新闻评分由 Financial-MCP-Agent 中的 score_financial_news 工具完成。
            # 此处不再重复加载旧路径中的模型，避免显存翻倍和硬编码路径失败。
            risk_model = risk_tokenizer = None
            sentiment_model = sentiment_tokenizer = None
            
            # 使用百度新闻搜索（更容易绕过反爬）
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            # 使用百度新闻搜索，而不是普通搜索
            search_url = f"https://www.baidu.com/s?tn=news&wd={encoded_query}&ie=utf-8"
            
            # 使用更完整的 headers 来模拟真实浏览器
            # 注意：不使用 br 压缩，避免解压问题
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',  # 移除 br，只使用 gzip 和 deflate
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
                'Referer': 'https://www.baidu.com/'
            }
            
            # 使用 Session 来保持 cookies
            session = requests.Session()
            session.headers.update(headers)
            
            # 先访问百度首页获取 cookies
            try:
                session.get('https://www.baidu.com', timeout=10)
            except:
                pass
            
            # 然后访问搜索页面
            response = session.get(search_url, timeout=15)
            response.raise_for_status()
            
            logger.info(f"获取到响应，状态码: {response.status_code}, 长度: {len(response.content)}")
            
            # 检查是否返回了验证页面
            response_text = response.text
            if '百度安全验证' in response_text or '安全验证' in response_text:
                logger.warning("百度返回了安全验证页面，尝试使用备用方法")
                # 尝试使用不同的搜索方式
                search_url = f"https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&tn=news&wd={encoded_query}"
                response = session.get(search_url, timeout=15)
                response_text = response.text
            
            # 使用 response.text 而不是 response.content，让 requests 自动处理编码
            # 如果 response.text 有问题，尝试手动指定编码
            try:
                soup = BeautifulSoup(response_text, 'html.parser')
            except Exception as e:
                logger.warning(f"使用text解析失败: {e}，尝试使用content")
                # 如果text解析失败，尝试使用content并指定编码
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            
            # 检查页面标题
            title_tag = soup.find('title')
            page_title = title_tag.get_text() if title_tag else '无标题'
            logger.info(f"页面标题: {page_title}")
            
            # 检查是否还是验证页面
            if '百度安全验证' in response.text or '安全验证' in response.text or 'timeout' in page_title.lower():
                logger.error("百度安全验证无法绕过，返回空结果")
                return "抱歉，百度搜索触发了安全验证，无法获取搜索结果。请稍后重试或使用其他搜索方式。"
            
            # 提取搜索结果 - 百度新闻搜索的 HTML 结构
            results = []
            
            # 方法1: 直接查找所有 h3 标签，然后过滤（更可靠的方法）
            # 百度新闻的 h3 标签通常有类名：news-title_xxx, c-title 等
            all_h3 = soup.find_all('h3')
            
            # 过滤出可能是新闻标题的 h3（有特定类名或包含链接）
            title_elements = []
            for h3 in all_h3:
                classes = h3.get('class', [])
                class_str = ' '.join(str(c) for c in classes) if classes else ''
                # 检查是否有新闻相关的类名，或者包含链接
                if ('title' in class_str.lower() or 'news' in class_str.lower() or 'c-title' in class_str) and h3.find('a'):
                    title_elements.append(h3)
            
            # 如果过滤后没有结果，使用所有h3
            if not title_elements:
                title_elements = [h3 for h3 in all_h3 if h3.find('a')]
            
            logger.info(f"找到 {len(all_h3)} 个h3标签，其中 {len(title_elements)} 个可能是新闻标题")
            
            for title_elem in title_elements[:top_k * 2]:  # 多找一些，因为会被过滤
                try:
                    # 查找标题中的链接
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        logger.debug(f"标题元素没有链接，跳过")
                        continue
                    
                    title = link_elem.get_text(strip=True)
                    link = link_elem.get('href', '')
                    
                    logger.debug(f"处理标题: {title[:50]}")
                    
                    # 过滤掉一些非新闻链接（如百度百科、官网等）
                    if not title or len(title) < 3:
                        logger.debug(f"标题太短，跳过: {title}")
                        continue
                    if any(skip in title for skip in ['官方网站', '百度百科', '移动官网']):
                        logger.debug(f"标题包含过滤词，跳过: {title}")
                        continue
                    
                    # 处理百度跳转链接
                    if link.startswith('/link?url='):
                        try:
                            import re
                            actual_url = re.search(r'url=([^&]+)', link)
                            if actual_url:
                                link = urllib.parse.unquote(actual_url.group(1))
                        except Exception as e:
                            logger.warning(f"解析跳转链接失败: {e}")
                    
                    # 查找摘要 - 在父元素中查找
                    parent = title_elem.find_parent()
                    abstract = ''
                    if parent:
                        # 查找摘要元素（常见的类名：c-abstract, c-span9, summary等）
                        abstract_elem = parent.find(['div', 'span'], class_=lambda x: x and (
                            'abstract' in str(x).lower() or 
                            'content' in str(x).lower() or 
                            'summary' in str(x).lower() or
                            'c-abstract' in str(x) or
                            'c-span' in str(x)
                        ) if x else False)
                        
                        if abstract_elem:
                            abstract = abstract_elem.get_text(strip=True)
                        else:
                            # 如果没找到，尝试查找所有文本内容作为摘要
                            all_text = parent.get_text(strip=True)
                            # 移除标题，保留剩余文本作为摘要
                            abstract = all_text.replace(title, '', 1).strip()[:200]
                    
                    # 获取完整文章内容
                    full_content = self._get_article_content(link) if link and link.startswith('http') else abstract
                    if not full_content:
                        full_content = abstract
                    
                    # 使用模型分析内容
                    risk_analysis = self._analyze_risk(full_content, risk_model, risk_tokenizer) if risk_model else "未分析"
                    sentiment_analysis = self._analyze_sentiment(full_content, sentiment_model, sentiment_tokenizer) if sentiment_model else "未分析"
                    
                    results.append({
                        'title': title,
                        'content': full_content,
                        'link': link,
                        'source': '百度新闻',
                        'date': '未知',
                        'risk': risk_analysis,
                        'sentiment': sentiment_analysis
                    })
                    
                    logger.info(f"成功提取新闻: {title[:50]}")
                    
                except Exception as e:
                    logger.warning(f"提取标题时出错: {e}")
                    continue
            
            # 方法2: 如果方法1没找到足够的新闻，尝试从结果容器中提取
            if len(results) < top_k:
                search_results = soup.find_all(['div', 'article'], class_=lambda x: x and (
                    'result' in str(x).lower() or 
                    'news' in str(x).lower() or 
                    'c-result' in str(x).lower()
                ) if x else False)
                
                for result_container in search_results[:top_k * 2]:  # 多找一些，因为可能会过滤
                    try:
                        # 提取标题
                        title_elem = result_container.find('h3')
                        if not title_elem:
                            continue
                        
                        link_elem = title_elem.find('a')
                        if not link_elem:
                            continue
                        
                        title = link_elem.get_text(strip=True)
                        link = link_elem.get('href', '')
                        
                        # 跳过已经添加的结果
                        if any(r['title'] == title for r in results):
                            continue
                        
                        # 过滤非新闻链接
                        if not title or len(title) < 3:
                            continue
                        if any(skip in title for skip in ['官方网站', '百度百科', '移动官网']):
                            continue
                        
                        # 处理百度跳转链接
                        if link.startswith('/link?url='):
                            try:
                                import re
                                actual_url = re.search(r'url=([^&]+)', link)
                                if actual_url:
                                    link = urllib.parse.unquote(actual_url.group(1))
                            except:
                                pass
                        
                        # 提取摘要
                        abstract_elem = result_container.find(['div', 'span'], class_=lambda x: x and (
                            'abstract' in str(x).lower() or 
                            'content' in str(x).lower() or 
                            'c-abstract' in str(x)
                        ) if x else False)
                        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ''
                        
                        # 获取完整文章内容
                        full_content = self._get_article_content(link) if link and link.startswith('http') else abstract
                        if not full_content:
                            full_content = abstract
                        
                        # 使用模型分析内容
                        risk_analysis = self._analyze_risk(full_content, risk_model, risk_tokenizer) if risk_model else "未分析"
                        sentiment_analysis = self._analyze_sentiment(full_content, sentiment_model, sentiment_tokenizer) if sentiment_model else "未分析"
                        
                        results.append({
                            'title': title,
                            'content': full_content,
                            'link': link,
                            'source': '百度新闻',
                            'date': '未知',
                            'risk': risk_analysis,
                            'sentiment': sentiment_analysis
                        })
                        
                        if len(results) >= top_k:
                            break
                            
                    except Exception as e:
                        logger.warning(f"提取搜索结果时出错: {e}")
                        continue
            
            if not results:
                return "未找到相关新闻。"
            
            output = "找到以下相关新闻：\n\n"
            
            for i, result in enumerate(results, 1):
                output += f"{i}. {result['title']}\n"
                output += f"   来源: {result['source']}\n"
                if result['content']:
                    content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                    output += f"   内容: {content_preview}\n"
                output += f"   风险分析: {result['risk']}\n"
                output += f"   情感分析: {result['sentiment']}\n"
                output += f"   链接: {result['link']}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"爬取新闻时出错: {e}")
            return f"爬取新闻时出错: {str(e)}"

    def _get_article_content(self, url: str) -> str:
        """
        获取文章的完整内容
        
        Args:
            url: 文章链接
            
        Returns:
            文章内容
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 尝试多个内容选择器
            content_selectors = [
                'article p',
                '.article-content p',
                '.story-content p',
                '.post-content p',
                '.entry-content p',
                'p',
                '.content p'
            ]
            
            content_parts = []
            for selector in content_selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and len(text) > 30:  # 只保留有意义的段落
                            content_parts.append(text)
                    break
            
            return ' '.join(content_parts)
            
        except Exception as e:
            logger.warning(f"获取文章内容时出错: {e}")
            return ""
    
    def _load_risk_model(self):
        """加载风险模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            
            risk_model_path = "/mnt/data/guyx/self-learn/Finance/qwen_risk_model"
            base_model_name = "/mnt/data/guyx/self-learn/Finance/Qwen"
            
            # 检查CUDA可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # 加载LoRA适配器
            risk_model = PeftModel.from_pretrained(base_model, risk_model_path)
            
            # 确保模型在正确的设备上
            if device == "cpu":
                risk_model = risk_model.to(device)
            
            logger.info("风险模型加载成功")
            return risk_model, tokenizer
            
        except Exception as e:
            logger.error(f"加载风险模型时出错: {e}")
            return None, None
    
    def _load_sentiment_model(self):
        """加载情感模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            
            sentiment_model_path = "/mnt/data/guyx/self-learn/Finance/qwen_sentiment_model"
            base_model_name = "/mnt/data/guyx/self-learn/Finance/Qwen"
            
            # 检查CUDA可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # 加载LoRA适配器
            sentiment_model = PeftModel.from_pretrained(base_model, sentiment_model_path)
            
            # 确保模型在正确的设备上
            if device == "cpu":
                sentiment_model = sentiment_model.to(device)
            
            logger.info("情感模型加载成功")
            return sentiment_model, tokenizer
            
        except Exception as e:
            logger.error(f"加载情感模型时出错: {e}")
            return None, None
    
    def _analyze_risk(self, content: str, model, tokenizer) -> str:
        """使用风险模型分析内容"""
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            
            import torch
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            # 构建风险评估提示词
            system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
            
            user_content = f"News to Stock Symbol -- STOCK: {content}"
            
            prompt = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant:"""
            
            # 编码输入并移动到正确的设备
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测的风险分数
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            # 尝试提取数字
            try:
                risk_score = int(assistant_response.split()[0])
                if 1 <= risk_score <= 5:
                    risk_map = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
                    return f"{risk_score} ({risk_map[risk_score]})"
            except:
                pass
            
            return "无法分析风险"
            
        except Exception as e:
            logger.error(f"风险分析时出错: {e}")
            return f"风险分析失败: {str(e)}"
    
    def _analyze_sentiment(self, content: str, model, tokenizer) -> str:
        """使用情感模型分析内容"""
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            
            import torch
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            # 构建情感分析提示词
            system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
            
            user_content = f"News to Stock Symbol -- STOCK: {content}"
            
            prompt = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant:"""
            
            # 编码输入并移动到正确的设备
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测的情感分数
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            # 尝试提取数字
            try:
                sentiment_score = int(assistant_response.split()[0])
                if 1 <= sentiment_score <= 5:
                    sentiment_map = {1: "负面", 2: "轻微负面", 3: "中性", 4: "正面", 5: "极正面"}
                    return f"{sentiment_score} ({sentiment_map[sentiment_score]})"
            except:
                pass
            
            return "无法分析情感"
            
        except Exception as e:
            logger.error(f"情感分析时出错: {e}")
            return f"情感分析失败: {str(e)}"
