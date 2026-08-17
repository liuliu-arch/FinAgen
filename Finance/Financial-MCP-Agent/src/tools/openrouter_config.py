import os
import time
from google import genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.llm_clients import LLMClientFactory

# 设置日志记录
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not model:
    model = "gemini-1.5-flash"
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# 初始化 Gemini 客户端
client = genai.Client(api_key=api_key)
logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
        logger.debug(f"请求内容: {contents}")
        logger.debug(f"请求配置: {config}")

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.debug(f"响应内容: {response.text[:500]}...")
        return response
    except Exception as e:
        error_msg = str(e)
        if "location" in error_msg.lower():
            # 使用红色感叹号和红色文字提示
            logger.info(f"\033[91m❗ Gemini API 地理位置限制错误: 请使用美国节点VPN后重试\033[0m")
            logger.error(f"详细错误: {error_msg}")
        elif "AFC is enabled" in error_msg:
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {error_msg}")
            time.sleep(5)
        else:
            logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        client_type="auto", api_key=None, base_url=None):
    """
    获取聊天完成结果，包含重试逻辑

    Args:
        messages: 消息列表，OpenAI 格式
        model: 模型名称（可选）
        max_retries: 最大重试次数
        initial_retry_delay: 初始重试延迟（秒）
        client_type: 客户端类型 ("auto", "gemini", "openai_compatible")
        api_key: API 密钥（可选，仅用于 OpenAI Compatible API）
        base_url: API 基础 URL（可选，仅用于 OpenAI Compatible API）

    Returns:
        str: 模型回答内容或 None（如果出错）
    """
    try:
        # 创建客户端
        client = LLMClientFactory.create_client(
            client_type=client_type,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        # 获取回答
        response = client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )

        # 检查响应格式，处理不同类型的返回值
        if isinstance(response, dict):
            # OpenAI 兼容 API 可能返回字典格式
            if 'choices' in response and len(response['choices']) > 0:
                if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    return response['choices'][0]['message']['content']
                elif 'text' in response['choices'][0]:
                    return response['choices'][0]['text']

        # 如果是字符串，直接返回
        if isinstance(response, str):
            return response

        # 其他类型的响应，尝试提取文本
        logger.warning(f"{WAIT_ICON} 未知响应格式，尝试提取文本: {type(response)}")
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content

        # 无法处理的响应格式
        logger.error(f"{ERROR_ICON} 无法从响应中提取文本: {response}")
        return str(response)
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
