"""MCP服务器配置模块。"""

from pathlib import Path


FINANCE_ROOT = Path(__file__).resolve().parents[3]
A_SHARE_MCP_ROOT = FINANCE_ROOT / "a-share-mcp-is-just-i-need"

SERVER_CONFIGS = {
    "a_share_mcp_v2": {  
        "command": "uv", 
        "args": [
            "run",  
            "--directory",
            str(A_SHARE_MCP_ROOT),
            "python",  #
            "mcp_server.py"  # MCP服务器脚本
        ],
        "transport": "stdio",
    }
}
