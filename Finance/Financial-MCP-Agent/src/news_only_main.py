"""Run the news agent when the current host cannot reach BaoStock TCP services."""

import argparse
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from src.agents.news_agent import news_agent
from src.utils.execution_logger import (
    finalize_execution_logger,
    initialize_execution_logger,
)


async def run():
    parser = argparse.ArgumentParser(description="Financial news-only Agent")
    parser.add_argument("--company", required=True)
    parser.add_argument("--stock", required=True)
    args = parser.parse_args()

    load_dotenv(override=True)
    os.environ["NEWS_ONLY_MODE"] = "true"
    execution_logger = initialize_execution_logger()
    now = datetime.now()
    state = {
        "messages": [],
        "data": {
            "query": f"分析{args.company}（{args.stock}）的最新新闻",
            "stock_code": args.stock,
            "company_name": args.company,
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time_info": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metadata": {},
    }

    try:
        result = await news_agent(state)
        data = result.get("data", {})
        if data.get("news_analysis_error"):
            raise RuntimeError(data["news_analysis_error"])
        print("\n===== 新闻分析结果 =====\n")
        print(data.get("news_analysis", "未生成新闻分析"))
        print(f"\n执行日志: {execution_logger.execution_dir}")
        finalize_execution_logger(success=True)
    except Exception as exc:
        finalize_execution_logger(success=False, error=str(exc))
        raise


if __name__ == "__main__":
    asyncio.run(run())
