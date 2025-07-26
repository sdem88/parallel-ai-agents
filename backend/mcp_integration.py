"""
🔌 MCP Integration Module
Интеграция Model Context Protocol серверов для расширения возможностей агентов
"""

import asyncio
import json
import httpx
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()

class MCPManager:
    """Менеджер MCP серверов"""
    
    def __init__(self):
        self.servers = {}
        self.tools = {}
        self.resources = {}
        self.initialize_mcp_servers()
    
    def initialize_mcp_servers(self):
        """Инициализация доступных MCP серверов"""
        # Конфигурация MCP серверов
        mcp_configs = {
            "filesystem": {
                "name": "File System Server",
                "description": "Работа с файловой системой",
                "tools": [
                    "read_file", "write_file", "list_directory", 
                    "create_directory", "search_files", "get_file_info"
                ],
                "enabled": True
            },
            "zapier": {
                "name": "Zapier Integration",
                "description": "Интеграция с тысячами сервисов через Zapier",
                "tools": [
                    "notion_create_page", "gmail_send_email", "slack_send_message",
                    "airtable_create_record", "google_sheets_append_row"
                ],
                "enabled": True
            },
            "terminal": {
                "name": "Terminal Controller", 
                "description": "Выполнение команд терминала",
                "tools": [
                    "execute_command", "get_current_directory", 
                    "change_directory", "list_directory"
                ],
                "enabled": True
            }
        }
        
        for server_id, config in mcp_configs.items():
            self.servers[server_id] = config
            for tool in config["tools"]:
                self.tools[f"{server_id}_{tool}"] = {
                    "server": server_id,
                    "tool": tool,
                    "description": f"{tool} from {config['name']}"
                }
        
        logger.info("MCP servers initialized", servers=list(self.servers.keys()))
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any], 
                              user_id: str = None) -> Dict[str, Any]:
        """Выполнение MCP инструмента"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_info = self.tools[tool_name]
        server_id = tool_info["server"]
        
        try:
            if server_id == "filesystem":
                return await self._execute_filesystem_tool(tool_info["tool"], parameters)
            elif server_id == "zapier":
                return await self._execute_zapier_tool(tool_info["tool"], parameters)
            elif server_id == "terminal":
                return await self._execute_terminal_tool(tool_info["tool"], parameters)
            else:
                raise ValueError(f"Unknown server: {server_id}")
                
        except Exception as e:
            logger.error("MCP tool execution failed", 
                        tool=tool_name, error=str(e), user_id=user_id)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _execute_filesystem_tool(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение файловых операций"""
        try:
            if tool == "read_file":
                # Безопасное чтение файлов (только в разрешенных директориях)
                file_path = params.get("path", "")
                if not self._is_safe_path(file_path):
                    raise ValueError("Access denied to this path")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return {
                    "success": True,
                    "content": content,
                    "file_path": file_path
                }
            
            elif tool == "write_file":
                file_path = params.get("path", "")
                content = params.get("content", "")
                
                if not self._is_safe_path(file_path):
                    raise ValueError("Access denied to this path")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "message": f"File written successfully to {file_path}",
                    "bytes_written": len(content.encode('utf-8'))
                }
            
            elif tool == "list_directory":
                import os
                dir_path = params.get("path", ".")
                
                if not self._is_safe_path(dir_path):
                    raise ValueError("Access denied to this path")
                
                items = []
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    items.append({
                        "name": item,
                        "type": "directory" if os.path.isdir(item_path) else "file",
                        "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                    })
                
                return {
                    "success": True,
                    "items": items,
                    "path": dir_path
                }
            
            elif tool == "search_files":
                import os
                import fnmatch
                
                search_path = params.get("path", ".")
                pattern = params.get("pattern", "*")
                
                if not self._is_safe_path(search_path):
                    raise ValueError("Access denied to this path")
                
                matches = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                            matches.append(os.path.join(root, file))
                
                return {
                    "success": True,
                    "matches": matches[:100],  # Ограничение результатов
                    "total_found": len(matches)
                }
            
            else:
                raise ValueError(f"Unknown filesystem tool: {tool}")
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": f"filesystem_{tool}"
            }
    
    async def _execute_zapier_tool(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение Zapier интеграций"""
        # В реальной реализации здесь был бы вызов Zapier API
        # Для демонстрации возвращаем заглушку
        
        zapier_responses = {
            "notion_create_page": {
                "success": True,
                "page_id": f"notion_page_{asyncio.get_event_loop().time()}",
                "message": "Page created in Notion successfully"
            },
            "gmail_send_email": {
                "success": True,
                "message_id": f"gmail_msg_{asyncio.get_event_loop().time()}",
                "message": "Email sent successfully"
            },
            "slack_send_message": {
                "success": True,
                "timestamp": asyncio.get_event_loop().time(),
                "message": "Message sent to Slack channel"
            },
            "airtable_create_record": {
                "success": True,
                "record_id": f"airtable_rec_{asyncio.get_event_loop().time()}",
                "message": "Record created in Airtable"
            },
            "google_sheets_append_row": {
                "success": True,
                "row_number": 42,
                "message": "Row appended to Google Sheets"
            }
        }
        
        if tool in zapier_responses:
            response = zapier_responses[tool].copy()
            response["parameters"] = params
            return response
        else:
            return {
                "success": False,
                "error": f"Unknown Zapier tool: {tool}"
            }
    
    async def _execute_terminal_tool(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение команд терминала (с ограничениями безопасности)"""
        import subprocess
        import os
        
        try:
            if tool == "execute_command":
                command = params.get("command", "")
                timeout = params.get("timeout", 30)
                
                # Список разрешенных команд (белый список)
                allowed_commands = [
                    "ls", "pwd", "date", "whoami", "echo", "cat", "head", "tail",
                    "wc", "grep", "find", "which", "python", "node", "npm", "pip"
                ]
                
                cmd_parts = command.split()
                if not cmd_parts or cmd_parts[0] not in allowed_commands:
                    raise ValueError(f"Command '{cmd_parts[0] if cmd_parts else command}' not allowed")
                
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.getcwd()
                )
                
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "command": command
                }
            
            elif tool == "get_current_directory":
                return {
                    "success": True,
                    "current_directory": os.getcwd()
                }
            
            elif tool == "list_directory":
                path = params.get("path", ".")
                items = os.listdir(path)
                return {
                    "success": True,
                    "items": items,
                    "path": path
                }
            
            else:
                raise ValueError(f"Unknown terminal tool: {tool}")
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "tool": f"terminal_{tool}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": f"terminal_{tool}"
            }
    
    def _is_safe_path(self, path: str) -> bool:
        """Проверка безопасности пути"""
        import os
        
        # Запрещенные пути
        forbidden_patterns = [
            "/etc/", "/var/", "/usr/", "/sys/", "/proc/",
            "C:\\Windows\\", "C:\\Program Files\\",
            "../", "..\\"
        ]
        
        abs_path = os.path.abspath(path)
        
        for pattern in forbidden_patterns:
            if pattern in abs_path:
                return False
        
        # Разрешить только в определенных директориях
        allowed_roots = [
            "/Users/sergeidemchuk/agents-system",
            "/tmp/",
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop")
        ]
        
        for root in allowed_roots:
            if abs_path.startswith(os.path.abspath(root)):
                return True
        
        return False
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Получить список доступных MCP инструментов"""
        tools_by_server = {}
        
        for tool_name, tool_info in self.tools.items():
            server = tool_info["server"]
            if server not in tools_by_server:
                tools_by_server[server] = {
                    "name": self.servers[server]["name"],
                    "description": self.servers[server]["description"],
                    "enabled": self.servers[server]["enabled"],
                    "tools": []
                }
            
            tools_by_server[server]["tools"].append({
                "name": tool_info["tool"],
                "full_name": tool_name,
                "description": tool_info["description"]
            })
        
        return tools_by_server
    
    async def enhance_agent_prompt(self, base_prompt: str, available_tools: List[str] = None) -> str:
        """Улучшение промпта агента с информацией о доступных MCP инструментах"""
        if not available_tools:
            available_tools = list(self.tools.keys())
        
        tools_description = "\n".join([
            f"- {tool}: {self.tools[tool]['description']}"
            for tool in available_tools
            if tool in self.tools
        ])
        
        enhanced_prompt = f"""{base_prompt}

ДОСТУПНЫЕ MCP ИНСТРУМЕНТЫ:
{tools_description}

Ты можешь использовать эти инструменты для выполнения задач. Когда нужно использовать инструмент, 
четко укажи его название и параметры в формате:

[TOOL: tool_name]
параметры: {{параметры в JSON формате}}
[/TOOL]

Например:
[TOOL: filesystem_read_file]
{{"path": "/path/to/file.txt"}}
[/TOOL]
"""
        
        return enhanced_prompt

# Глобальный экземпляр MCP менеджера
mcp_manager = MCPManager()

# Функции для интеграции с основной системой
async def execute_enhanced_task(agent, task_description: str, user_id: str = None) -> Dict[str, Any]:
    """Выполнение задачи с поддержкой MCP инструментов"""
    
    # Проверить, есть ли в описании задачи запросы на использование инструментов
    if "[TOOL:" in task_description:
        return await _process_task_with_tools(agent, task_description, user_id)
    else:
        # Обычное выполнение задачи
        return await agent.execute_task_original(task_description, user_id)

async def _process_task_with_tools(agent, task_description: str, user_id: str = None) -> Dict[str, Any]:
    """Обработка задачи с MCP инструментами"""
    import re
    
    # Найти все вызовы инструментов в описании задачи
    tool_pattern = r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]'
    tool_matches = re.findall(tool_pattern, task_description, re.DOTALL)
    
    results = []
    
    for tool_name, params_str in tool_matches:
        try:
            # Парсинг параметров JSON
            params = json.loads(params_str.strip())
            
            # Выполнение MCP инструмента
            tool_result = await mcp_manager.execute_mcp_tool(tool_name, params, user_id)
            results.append({
                "tool": tool_name,
                "parameters": params,
                "result": tool_result
            })
            
        except json.JSONDecodeError:
            results.append({
                "tool": tool_name,
                "error": "Invalid JSON parameters",
                "parameters_raw": params_str
            })
        except Exception as e:
            results.append({
                "tool": tool_name,
                "error": str(e),
                "parameters_raw": params_str
            })
    
    # Если есть результаты MCP инструментов, добавить их в контекст для агента
    if results:
        enhanced_task = f"""{task_description}

РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ ИНСТРУМЕНТОВ:
{json.dumps(results, indent=2, ensure_ascii=False)}

Проанализируй результаты и дай финальный ответ на основе полученных данных."""
        
        return await agent.execute_task_original(enhanced_task, user_id)
    else:
        return await agent.execute_task_original(task_description, user_id)

def get_mcp_tools_for_api():
    """Получить информацию о MCP инструментах для API"""
    return mcp_manager.get_available_tools()

# Примеры использования MCP инструментов в промптах
MCP_EXAMPLES = {
    "file_operations": """
Пример работы с файлами:
[TOOL: filesystem_read_file]
{"path": "/path/to/document.txt"}
[/TOOL]

[TOOL: filesystem_write_file]
{"path": "/path/to/output.txt", "content": "Новый контент файла"}
[/TOOL]
""",
    
    "zapier_integrations": """
Пример интеграций через Zapier:
[TOOL: zapier_notion_create_page]
{"title": "Новая страница", "content": "Содержимое страницы"}
[/TOOL]

[TOOL: zapier_gmail_send_email]
{"to": "example@email.com", "subject": "Тема письма", "body": "Текст письма"}
[/TOOL]
""",
    
    "terminal_commands": """
Пример выполнения команд:
[TOOL: terminal_execute_command]
{"command": "ls -la", "timeout": 10}
[/TOOL]

[TOOL: terminal_get_current_directory]
{}
[/TOOL]
"""
}

logger.info("MCP integration module loaded successfully")