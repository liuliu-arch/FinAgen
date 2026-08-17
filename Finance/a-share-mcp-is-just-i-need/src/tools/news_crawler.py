"""
新闻爬虫工具模块
提供新闻搜索和爬取功能
"""

import logging
from typing import List, Dict
from mcp.server.fastmcp import FastMCP
from ..data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)

def register_news_crawler_tools(app: FastMCP, data_source: FinancialDataSource):
    """
    注册新闻爬虫工具
    
    Args:
        app: FastMCP应用实例
        data_source: 数据源实例
    """
    
    @app.tool()
    def crawl_news(query: str, top_k: int = 10) -> str:
        """
        爬取相关新闻
        
        使用百度搜索爬取与查询词相关的新闻文章，并返回格式化的结果。
        
        Args:
            query: 搜索查询词，如"嘉友国际"、"人工智能投资"等
            top_k: 返回的新闻数量，默认为10条
            
        Returns:
            格式化的新闻结果字符串，包含标题、内容摘要、链接等信息
            
        Example:
            >>> crawl_news("嘉友国际", 5)
            "找到以下相关新闻：
            
            1. 嘉友国际发布2024年第一季度财报
               来源: 百度搜索
               内容: 嘉友国际今日发布2024年第一季度财报，营收同比增长15%...
               链接: https://example.com/news/123
            "
        """
        try:
            logger.info(f"开始爬取新闻，查询词: {query}, 数量: {top_k}")
            result = data_source.crawl_news(query, top_k)
            logger.info(f"新闻爬取完成，返回结果长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"爬取新闻时出错: {e}")
            return f"爬取新闻时出错: {str(e)}"
    
    