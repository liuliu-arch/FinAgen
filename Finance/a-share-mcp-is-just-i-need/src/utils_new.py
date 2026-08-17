import baostock as bs
import os
import sys
import logging
import pandas as pd
import threading
from contextlib import contextmanager
from typing import List, Optional, Callable, Any
from .data_source_interface import LoginError, DataSourceError, NoDataFoundError

# --- 日志设置 ---
def setup_logging(level=logging.INFO):
    """配置应用程序的基本日志记录"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 可选地静音依赖项的日志，如果它们太冗长的话
    # logging.getLogger("mcp").setLevel(logging.WARNING)

# 获取此模块的日志记录器实例（可选，但这是好习惯）
logger = logging.getLogger(__name__)

# --- Baostock 线程安全登录管理器 ---
class BaostockSessionManager:
    """
    线程安全的 Baostock 会话管理器
    
    使用引用计数来管理登录状态：
    - 第一个使用者会触发登录
    - 后续使用者只增加引用计数
    - 最后一个使用者退出时才登出
    
    这解决了并发环境下多个协程同时使用 baostock 导致的竞态条件问题。
    """
    
    def __init__(self):
        self._lock = threading.Lock()  # 线程锁，保护共享状态
        self._ref_count = 0            # 引用计数
        self._logged_in = False        # 登录状态
    
    def _suppress_stdout(self):
        """重定向标准输出到 /dev/null，抑制 baostock 的打印信息"""
        self._original_stdout_fd = sys.stdout.fileno()
        self._saved_stdout_fd = os.dup(self._original_stdout_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, self._original_stdout_fd)
        os.close(devnull_fd)
    
    def _restore_stdout(self):
        """恢复标准输出"""
        os.dup2(self._saved_stdout_fd, self._original_stdout_fd)
        os.close(self._saved_stdout_fd)
    
    def acquire(self):
        """
        获取 baostock 会话
        
        如果尚未登录，则执行登录；否则只增加引用计数。
        线程安全。
        """
        with self._lock:
            self._ref_count += 1
            logger.debug(f"Baostock session acquired, ref_count={self._ref_count}")
            
            if not self._logged_in:
                # 第一个使用者，执行登录
                self._suppress_stdout()
                try:
                    logger.debug("Attempting Baostock login...")
                    lg = bs.login()
                    logger.debug(f"Login result: code={lg.error_code}, msg={lg.error_msg}")
                finally:
                    self._restore_stdout()
                
                if lg.error_code != '0':
                    self._ref_count -= 1  # 登录失败，回滚引用计数
                    logger.error(f"Baostock login failed: {lg.error_msg}")
                    raise LoginError(f"Baostock login failed: {lg.error_msg}")
                
                self._logged_in = True
                logger.info("Baostock login successful (first session).")
            else:
                logger.debug(f"Baostock already logged in, reusing session (ref_count={self._ref_count}).")
    
    def release(self):
        """
        释放 baostock 会话
        
        减少引用计数，当引用计数归零时执行登出。
        线程安全。
        """
        with self._lock:
            self._ref_count -= 1
            logger.debug(f"Baostock session released, ref_count={self._ref_count}")
            
            if self._ref_count <= 0 and self._logged_in:
                # 最后一个使用者，执行登出
                self._suppress_stdout()
                try:
                    logger.debug("Attempting Baostock logout...")
                    bs.logout()
                    logger.debug("Logout completed.")
                finally:
                    self._restore_stdout()
                
                self._logged_in = False
                self._ref_count = 0  # 确保不会变成负数
                logger.info("Baostock logout successful (last session).")


# 全局会话管理器实例
_session_manager = BaostockSessionManager()


# --- Baostock上下文管理器 ---
@contextmanager
def baostock_login_context():
    """
    上下文管理器，处理Baostock登录和登出，抑制标准输出消息
    
    这是一个线程安全的实现，使用引用计数来管理登录状态：
    - 支持多个协程/线程并发使用
    - 第一个使用者登录，最后一个使用者登出
    - 避免了并发环境下的"用户未登录"错误
    """
    _session_manager.acquire()
    try:
        yield  # API调用在这里进行
    finally:
        _session_manager.release()

# --- 通用数据获取函数 ---

def fetch_financial_data(
    bs_query_func: Callable,
    data_type_name: str,
    code: str,
    year: str,
    quarter: int,
    **kwargs
) -> pd.DataFrame:
    """
    通用的财务数据获取函数
    
    参数:
        bs_query_func: Baostock的具体查询函数
        data_type_name: 数据类型名称，用于日志记录和错误信息
        code: 股票代码（如"sz.000001"）
        year: 年份（如"2023"）
        quarter: 季度（1-4）
        **kwargs: 额外参数
        
    返回:
        包含财务数据的pandas DataFrame
        
    异常:
        LoginError: 登录失败
        NoDataFoundError: 未找到数据
        DataSourceError: 数据源错误
    """
    logger.info(
        f"Fetching {data_type_name} data for {code}, year={year}, quarter={quarter}")
    
    try:
        # 使用登录上下文管理器确保API连接正常
        with baostock_login_context():
            # 调用传入的Baostock查询函数，所有财务数据函数都使用相同的参数格式
            rs = bs_query_func(code=code, year=year, quarter=quarter, **kwargs)

            # 检查API返回的错误码，'0'表示成功
            if rs.error_code != '0':
                logger.error(
                    f"Baostock API error ({data_type_name}) for {code}: {rs.error_msg} (code: {rs.error_code})")
                
                # 区分"无数据"和"API错误"两种情况
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    # 10002是常见的无数据错误码
                    raise NoDataFoundError(
                        f"No {data_type_name} data found for {code}, {year}Q{quarter}. Baostock msg: {rs.error_msg}")
                else:
                    # 其他API错误
                    raise DataSourceError(
                        f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})")

            # 遍历结果集，收集所有数据行
            data_list = []
            while rs.next():  # rs.next()返回True表示还有数据
                data_list.append(rs.get_row_data())  # 获取当前行的数据

            # 检查是否为空结果集
            if not data_list:
                logger.warning(
                    f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set from Baostock).")
                raise NoDataFoundError(
                    f"No {data_type_name} data found for {code}, {year}Q{quarter} (empty result set).")

            # 将数据转换为pandas DataFrame，使用rs.fields作为列名
            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {data_type_name} records for {code}, {year}Q{quarter}.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        # 已知异常直接重新抛出，不做额外处理
        logger.warning(
            f"Caught known error fetching {data_type_name} data for {code}: {type(e).__name__}")
        raise e
    except Exception as e:
        # 未预期的异常，记录详细信息并包装为DataSourceError
        logger.exception(
            f"Unexpected error fetching {data_type_name} data for {code}: {e}")
        raise DataSourceError(
            f"Unexpected error fetching {data_type_name} data: {e}")


def fetch_index_constituent_data(
    bs_query_func: Callable,
    index_name: str,
    date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    通用的指数成分股数据获取函数
    
    参数:
        bs_query_func: Baostock的具体指数查询函数
        index_name: 指数名称，用于日志记录和错误信息
        date: 查询日期，可选参数，默认获取最新数据
        **kwargs: 额外参数
        
    返回:
        包含指数成分股信息的pandas DataFrame
        
    异常:
        LoginError: 登录失败
        NoDataFoundError: 未找到数据
        DataSourceError: 数据源错误
    """
    logger.info(
        f"Fetching {index_name} constituents for date={date or 'latest'}")
    
    try:
        # 使用登录上下文管理器确保API连接正常
        with baostock_login_context():
            # date参数是可选的，如果不提供则默认获取最新数据
            rs = bs_query_func(date=date, **kwargs)

            # 检查API返回的错误码，'0'表示成功
            if rs.error_code != '0':
                logger.error(
                    f"Baostock API error ({index_name} Constituents) for date {date}: {rs.error_msg} (code: {rs.error_code})")
                
                # 区分"无数据"和"API错误"两种情况
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    # 10002是常见的无数据错误码
                    raise NoDataFoundError(
                        f"No {index_name} constituent data found for date {date}. Baostock msg: {rs.error_msg}")
                else:
                    # 其他API错误
                    raise DataSourceError(
                        f"Baostock API error fetching {index_name} constituents: {rs.error_msg} (code: {rs.error_code})")

            # 遍历结果集，收集所有成分股数据行
            data_list = []
            while rs.next():  # rs.next()返回True表示还有数据
                data_list.append(rs.get_row_data())  # 获取当前行的数据

            # 检查是否为空结果集
            if not data_list:
                logger.warning(
                    f"No {index_name} constituent data found for date {date} (empty result set).")
                raise NoDataFoundError(
                    f"No {index_name} constituent data found for date {date} (empty result set).")

            # 将数据转换为pandas DataFrame，使用rs.fields作为列名
            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {index_name} constituents for date {date or 'latest'}.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        # 已知异常直接重新抛出，不做额外处理
        logger.warning(
            f"Caught known error fetching {index_name} constituents for date {date}: {type(e).__name__}")
        raise e
    except Exception as e:
        # 未预期的异常，记录详细信息并包装为DataSourceError
        logger.exception(
            f"Unexpected error fetching {index_name} constituents for date {date}: {e}")
        raise DataSourceError(
            f"Unexpected error fetching {index_name} constituents for date {date}: {e}")


def fetch_macro_data(
    bs_query_func: Callable,
    data_type_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    通用的宏观经济数据获取函数
    
    参数:
        bs_query_func: Baostock的具体宏观数据查询函数
        data_type_name: 数据类型名称，用于日志记录和错误信息
        start_date: 查询开始日期，可选参数，使用默认范围
        end_date: 查询结束日期，可选参数，使用默认范围
        **kwargs: 额外参数，如yearType等，支持不同API的特殊需求
        
    返回:
        包含宏观经济数据的pandas DataFrame
        
    异常:
        LoginError: 登录失败
        NoDataFoundError: 未找到数据
        DataSourceError: 数据源错误
    """
    # 构建日志消息，显示查询时间范围和额外参数
    date_range_log = f"from {start_date or 'default'} to {end_date or 'default'}"
    kwargs_log = f", extra_args={kwargs}" if kwargs else ""
    logger.info(f"Fetching {data_type_name} data {date_range_log}{kwargs_log}")
    
    try:
        # 使用登录上下文管理器确保API连接正常
        with baostock_login_context():
            # 调用传入的Baostock查询函数，传递时间范围和额外参数
            rs = bs_query_func(start_date=start_date,
                               end_date=end_date, **kwargs)

            # 检查API返回的错误码，'0'表示成功
            if rs.error_code != '0':
                logger.error(
                    f"Baostock API error ({data_type_name}): {rs.error_msg} (code: {rs.error_code})")
                
                # 区分"无数据"和"API错误"两种情况
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    # 10002是常见的无数据错误码
                    raise NoDataFoundError(
                        f"No {data_type_name} data found for the specified criteria. Baostock msg: {rs.error_msg}")
                else:
                    # 其他API错误
                    raise DataSourceError(
                        f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})")

            # 遍历结果集，收集所有宏观经济数据行
            data_list = []
            while rs.next():  # rs.next()返回True表示还有数据
                data_list.append(rs.get_row_data())  # 获取当前行的数据

            # 检查是否为空结果集
            if not data_list:
                logger.warning(
                    f"No {data_type_name} data found for the specified criteria (empty result set).")
                raise NoDataFoundError(
                    f"No {data_type_name} data found for the specified criteria (empty result set).")

            # 将数据转换为pandas DataFrame，使用rs.fields作为列名
            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {data_type_name} records.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        # 已知异常直接重新抛出，不做额外处理
        logger.warning(
            f"Caught known error fetching {data_type_name} data: {type(e).__name__}")
        raise e
    except Exception as e:
        # 未预期的异常，记录详细信息并包装为DataSourceError
        logger.exception(
            f"Unexpected error fetching {data_type_name} data: {e}")
        raise DataSourceError(
            f"Unexpected error fetching {data_type_name} data: {e}")


def fetch_generic_data(
    bs_query_func: Callable,
    data_type_name: str,
    **kwargs
) -> pd.DataFrame:
    """
    通用的数据获取函数，适用于各种Baostock API调用
    
    参数:
        bs_query_func: Baostock的具体查询函数
        data_type_name: 数据类型名称，用于日志记录和错误信息
        **kwargs: 传递给查询函数的参数
        
    返回:
        包含数据的pandas DataFrame
        
    异常:
        LoginError: 登录失败
        NoDataFoundError: 未找到数据
        DataSourceError: 数据源错误
    """
    # 构建日志消息
    kwargs_log = f" with args: {kwargs}" if kwargs else ""
    logger.info(f"Fetching {data_type_name} data{kwargs_log}")
    
    try:
        # 使用登录上下文管理器确保API连接正常
        with baostock_login_context():
            # 调用传入的Baostock查询函数
            rs = bs_query_func(**kwargs)

            # 检查API返回的错误码，'0'表示成功
            if rs.error_code != '0':
                logger.error(
                    f"Baostock API error ({data_type_name}): {rs.error_msg} (code: {rs.error_code})")
                
                # 区分"无数据"和"API错误"两种情况
                if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                    # 10002是常见的无数据错误码
                    raise NoDataFoundError(
                        f"No {data_type_name} data found for the specified criteria. Baostock msg: {rs.error_msg}")
                else:
                    # 其他API错误
                    raise DataSourceError(
                        f"Baostock API error fetching {data_type_name} data: {rs.error_msg} (code: {rs.error_code})")

            # 遍历结果集，收集所有数据行
            data_list = []
            while rs.next():  # rs.next()返回True表示还有数据
                data_list.append(rs.get_row_data())  # 获取当前行的数据

            # 检查是否为空结果集
            if not data_list:
                logger.warning(
                    f"No {data_type_name} data found for the specified criteria (empty result set).")
                raise NoDataFoundError(
                    f"No {data_type_name} data found for the specified criteria (empty result set).")

            # 将数据转换为pandas DataFrame，使用rs.fields作为列名
            result_df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(
                f"Retrieved {len(result_df)} {data_type_name} records.")
            return result_df

    except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
        # 已知异常直接重新抛出，不做额外处理
        logger.warning(
            f"Caught known error fetching {data_type_name} data: {type(e).__name__}")
        raise e
    except Exception as e:
        # 未预期的异常，记录详细信息并包装为DataSourceError
        logger.exception(
            f"Unexpected error fetching {data_type_name} data: {e}")
        raise DataSourceError(
            f"Unexpected error fetching {data_type_name} data: {e}")


def format_fields(fields: Optional[List[str]], default_fields: List[str]) -> str:
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
    # 如果未指定字段或字段列表为空，则使用默认字段
    if fields is None or not fields:
        logger.debug(
            f"No specific fields requested, using defaults: {default_fields}")
        return ",".join(default_fields)
    
    # 基本验证：确保所有请求字段都是字符串类型
    if not all(isinstance(f, str) for f in fields):
        raise ValueError("All items in the fields list must be strings.")
    
    logger.debug(f"Using requested fields: {fields}")
    return ",".join(fields)