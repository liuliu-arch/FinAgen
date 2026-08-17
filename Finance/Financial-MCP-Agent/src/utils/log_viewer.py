"""
日志查看器 - 用于查看和分析执行日志
提供多种查看方式：最新执行、按时间范围、按执行ID等
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class LogViewer:
    """执行日志查看器"""

    def __init__(self, base_log_dir: str = "logs"):
        """
        初始化日志查看器

        Args:
            base_log_dir: 基础日志目录
        """
        self.base_log_dir = Path(base_log_dir)

    def list_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出最近的执行记录

        Args:
            limit: 返回的记录数量限制

        Returns:
            执行记录列表
        """
        executions = []

        if not self.base_log_dir.exists():
            return executions

        # 获取所有执行目录
        execution_dirs = [d for d in self.base_log_dir.iterdir() if d.is_dir()]

        # 按创建时间排序
        execution_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        for exec_dir in execution_dirs[:limit]:
            execution_info_file = exec_dir / "execution_info.json"
            if execution_info_file.exists():
                try:
                    with open(execution_info_file, 'r', encoding='utf-8') as f:
                        execution_info = json.load(f)

                    # 添加目录路径信息
                    execution_info["log_directory"] = str(exec_dir)
                    executions.append(execution_info)
                except Exception as e:
                    print(f"Error reading {execution_info_file}: {e}")

        return executions

    def get_execution_details(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定执行的详细信息

        Args:
            execution_id: 执行ID

        Returns:
            执行详细信息
        """
        execution_dir = self.base_log_dir / execution_id
        if not execution_dir.exists():
            return None

        details = {}

        # 读取执行信息
        execution_info_file = execution_dir / "execution_info.json"
        if execution_info_file.exists():
            with open(execution_info_file, 'r', encoding='utf-8') as f:
                details["execution_info"] = json.load(f)

        # 读取agent执行信息
        agents_dir = execution_dir / "agents"
        if agents_dir.exists():
            details["agents"] = {}
            for agent_file in agents_dir.glob("*_execution.json"):
                agent_name = agent_file.stem.replace("_execution", "")
                with open(agent_file, 'r', encoding='utf-8') as f:
                    details["agents"][agent_name] = json.load(f)

        # 读取LLM交互信息
        llm_dir = execution_dir / "llm_interactions"
        if llm_dir.exists():
            details["llm_interactions"] = []
            for llm_file in llm_dir.glob("*.json"):
                with open(llm_file, 'r', encoding='utf-8') as f:
                    details["llm_interactions"].append(json.load(f))

        # 读取工具使用信息
        tools_dir = execution_dir / "tools"
        if tools_dir.exists():
            details["tool_usage"] = []
            for tool_file in tools_dir.glob("*.jsonl"):
                with open(tool_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            details["tool_usage"].append(json.loads(line))

        # 读取报告信息
        reports_dir = execution_dir / "reports"
        if reports_dir.exists():
            report_info_file = reports_dir / "final_report_info.json"
            if report_info_file.exists():
                with open(report_info_file, 'r', encoding='utf-8') as f:
                    details["report_info"] = json.load(f)

        return details

    def print_execution_summary(self, execution_info: Dict[str, Any]):
        """打印执行摘要"""
        print(f"\n{'='*60}")
        print(f"执行ID: {execution_info.get('execution_id', 'Unknown')}")
        print(f"开始时间: {execution_info.get('start_time', 'Unknown')}")
        print(f"结束时间: {execution_info.get('end_time', 'Running...')}")

        if 'total_execution_time_seconds' in execution_info:
            print(
                f"总执行时间: {execution_info['total_execution_time_seconds']:.2f} 秒")

        status = execution_info.get('status', 'Unknown')
        status_icon = "✅" if execution_info.get('success', False) else "❌"
        print(f"执行状态: {status_icon} {status}")

        if 'environment' in execution_info:
            env = execution_info['environment']['environment_variables']
            print(f"使用模型: {env.get('OPENAI_COMPATIBLE_MODEL', 'Unknown')}")

        if 'summary' in execution_info:
            summary = execution_info['summary']
            print(f"\n执行统计:")
            print(f"  - Agent数量: {len(summary.get('agents_executed', []))}")
            print(f"  - LLM交互次数: {summary.get('llm_interactions_count', 0)}")
            print(f"  - 工具使用次数: {summary.get('tools_used_count', 0)}")
            print(f"  - 创建文件数: {summary.get('total_files_created', 0)}")

        if execution_info.get('error'):
            print(f"\n❌ 错误信息: {execution_info['error']}")

        print(f"日志目录: {execution_info.get('log_directory', 'Unknown')}")

    def print_agent_details(self, agents_info: Dict[str, Any]):
        """打印agent执行详情"""
        print(f"\n{'='*60}")
        print("AGENT 执行详情")
        print(f"{'='*60}")

        for agent_name, agent_data in agents_info.items():
            status_icon = "✅" if agent_data.get('success', False) else "❌"
            print(f"\n{status_icon} {agent_name.upper()}")
            print(f"  开始时间: {agent_data.get('start_time', 'Unknown')}")
            print(f"  结束时间: {agent_data.get('end_time', 'Unknown')}")

            if 'execution_time_seconds' in agent_data:
                print(f"  执行时间: {agent_data['execution_time_seconds']:.2f} 秒")

            if agent_data.get('error'):
                print(f"  ❌ 错误: {agent_data['error']}")

            # 显示输出预览
            output_data = agent_data.get('output_data', {})
            for key, value in output_data.items():
                if key.endswith('_preview') or key.endswith('_length'):
                    print(f"  {key}: {value}")

    def print_llm_interactions(self, llm_interactions: List[Dict[str, Any]]):
        """打印LLM交互详情"""
        print(f"\n{'='*60}")
        print("LLM 交互详情")
        print(f"{'='*60}")

        for interaction in llm_interactions:
            print(
                f"\n🤖 {interaction.get('agent_name', 'Unknown')} - {interaction.get('interaction_type', 'Unknown')}")
            print(f"  时间: {interaction.get('timestamp', 'Unknown')}")
            print(
                f"  模型: {interaction.get('model_config', {}).get('model', 'Unknown')}")
            print(
                f"  执行时间: {interaction.get('performance', {}).get('execution_time_seconds', 0):.2f} 秒")

            input_info = interaction.get('input', {})
            print(f"  输入消息数: {input_info.get('message_count', 0)}")
            print(f"  输入长度: {input_info.get('total_input_length', 0)} 字符")

            output_info = interaction.get('output', {})
            print(f"  输出长度: {output_info.get('content_length', 0)} 字符")

    def print_tool_usage(self, tool_usage: List[Dict[str, Any]]):
        """打印工具使用详情"""
        if not tool_usage:
            return

        print(f"\n{'='*60}")
        print("工具使用详情")
        print(f"{'='*60}")

        for tool_log in tool_usage:
            status_icon = "✅" if tool_log.get('success', True) else "❌"
            print(
                f"\n{status_icon} {tool_log.get('tool_name', 'Unknown')} (by {tool_log.get('agent_name', 'Unknown')})")
            print(f"  时间: {tool_log.get('timestamp', 'Unknown')}")
            print(f"  执行时间: {tool_log.get('execution_time_seconds', 0):.2f} 秒")

            if tool_log.get('error'):
                print(f"  ❌ 错误: {tool_log['error']}")

    def show_execution(self, execution_id: str, show_details: bool = True):
        """显示特定执行的完整信息"""
        details = self.get_execution_details(execution_id)
        if not details:
            print(f"❌ 未找到执行ID: {execution_id}")
            return

        # 显示执行摘要
        if 'execution_info' in details:
            self.print_execution_summary(details['execution_info'])

        if not show_details:
            return

        # 显示agent详情
        if 'agents' in details:
            self.print_agent_details(details['agents'])

        # 显示LLM交互详情
        if 'llm_interactions' in details:
            self.print_llm_interactions(details['llm_interactions'])

        # 显示工具使用详情
        if 'tool_usage' in details:
            self.print_tool_usage(details['tool_usage'])

        # 显示报告信息
        if 'report_info' in details:
            print(f"\n{'='*60}")
            print("报告信息")
            print(f"{'='*60}")
            report_info = details['report_info']
            print(f"报告路径: {report_info.get('report_path', 'Unknown')}")
            print(f"报告长度: {report_info.get('report_length', 0)} 字符")
            print(f"生成时间: {report_info.get('timestamp', 'Unknown')}")

    def show_recent_executions(self, limit: int = 5):
        """显示最近的执行记录"""
        executions = self.list_executions(limit)

        if not executions:
            print("❌ 未找到任何执行记录")
            return

        print(f"\n📊 最近 {len(executions)} 次执行记录:")
        print(f"{'='*80}")

        for i, execution in enumerate(executions, 1):
            print(f"\n{i}. {execution.get('execution_id', 'Unknown')}")
            status_icon = "✅" if execution.get('success', False) else "❌"
            print(f"   状态: {status_icon} {execution.get('status', 'Unknown')}")
            print(f"   时间: {execution.get('start_time', 'Unknown')}")

            if 'total_execution_time_seconds' in execution:
                print(
                    f"   耗时: {execution['total_execution_time_seconds']:.2f} 秒")

            if 'environment' in execution:
                env = execution['environment']['environment_variables']
                print(
                    f"   模型: {env.get('OPENAI_COMPATIBLE_MODEL', 'Unknown')}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="执行日志查看器")
    parser.add_argument("--list", "-l", action="store_true", help="列出最近的执行记录")
    parser.add_argument("--show", "-s", type=str, help="显示特定执行ID的详细信息")
    parser.add_argument("--limit", type=int, default=5, help="列出记录的数量限制")
    parser.add_argument(
        "--summary-only", action="store_true", help="只显示摘要，不显示详细信息")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录路径")

    args = parser.parse_args()

    viewer = LogViewer(args.log_dir)

    if args.show:
        viewer.show_execution(args.show, not args.summary_only)
    elif args.list:
        viewer.show_recent_executions(args.limit)
    else:
        # 默认显示最近的执行记录
        viewer.show_recent_executions(args.limit)


if __name__ == "__main__":
    main()
