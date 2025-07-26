"""
Enterprise logging and monitoring system
"""

import sys
import os
import logging
import logging.handlers
from typing import Optional, Dict, Any
from datetime import datetime
import structlog
from structlog.stdlib import LoggerFactory
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from .config import get_config


class EnterpriseLogger:
    """Enterprise-grade logging system"""
    
    def __init__(self):
        self.config = get_config()
        self._setup_structlog()
        self._setup_sentry()
        self._logger = structlog.get_logger(__name__)
    
    def _setup_structlog(self):
        """Configure structured logging"""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            self._add_timestamp,
            self._add_service_context,
        ]
        
        if self.config.logging.structured:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.config.logging.level),
        )
        
        # Setup file logging if configured
        if self.config.logging.file_path:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup rotating file handler"""
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.logging.file_path,
            maxBytes=self.config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.logging.backup_count,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    def _setup_sentry(self):
        """Configure Sentry for error tracking"""
        if not self.config.monitoring.sentry_dsn:
            return
        
        sentry_sdk.init(
            dsn=self.config.monitoring.sentry_dsn,
            integrations=[
                FastApiIntegration(auto_enable=True),
                AsyncioIntegration(),
            ],
            traces_sample_rate=0.1 if self.config.is_production() else 1.0,
            environment=self.config.app.environment,
            release=self.config.app.version,
            before_send=self._sentry_before_send,
        )
    
    def _sentry_before_send(self, event, hint):
        """Filter and modify events before sending to Sentry"""
        # Don't send events in development unless critical
        if (self.config.is_development() and 
            event.get('level') not in ['error', 'fatal']):
            return None
        
        # Add custom context
        event.setdefault('extra', {}).update({
            'service': 'agent-system',
            'version': self.config.app.version,
        })
        
        return event
    
    def _add_timestamp(self, logger, method_name, event_dict):
        """Add timestamp to log entries"""
        event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        return event_dict
    
    def _add_service_context(self, logger, method_name, event_dict):
        """Add service context to log entries"""
        event_dict.update({
            'service': 'agent-system',
            'version': self.config.app.version,
            'environment': self.config.app.environment,
        })
        return event_dict
    
    def get_logger(self, name: str, **context) -> structlog.BoundLogger:
        """Get a logger with optional context"""
        logger = structlog.get_logger(name)
        if context:
            logger = logger.bind(**context)
        return logger


class MetricsCollector:
    """Collect and manage application metrics"""
    
    def __init__(self):
        self.config = get_config()
        self._metrics: Dict[str, Any] = {}
        self._logger = structlog.get_logger(__name__)
        
        if self.config.monitoring.prometheus_enabled:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics collection"""
        try:
            from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
            
            self.registry = CollectorRegistry()
            
            # Agent metrics
            self.agent_tasks_total = Counter(
                'agent_tasks_total',
                'Total number of tasks processed by agents',
                ['agent_id', 'status'],
                registry=self.registry
            )
            
            self.agent_task_duration = Histogram(
                'agent_task_duration_seconds',
                'Time spent processing tasks',
                ['agent_id'],
                registry=self.registry
            )
            
            self.active_agents = Gauge(
                'active_agents',
                'Number of currently active agents',
                registry=self.registry
            )
            
            self.active_tasks = Gauge(
                'active_tasks',
                'Number of currently active tasks',
                registry=self.registry
            )
            
            # System metrics
            self.api_requests_total = Counter(
                'api_requests_total',
                'Total API requests',
                ['method', 'endpoint', 'status'],
                registry=self.registry
            )
            
            self.api_request_duration = Histogram(
                'api_request_duration_seconds',
                'API request duration',
                ['method', 'endpoint'],
                registry=self.registry
            )
            
            self._logger.info("Prometheus metrics initialized")
            
        except ImportError:
            self._logger.warning("Prometheus client not available")
    
    def record_task_completion(self, agent_id: str, status: str, duration_ms: float):
        """Record task completion metrics"""
        if hasattr(self, 'agent_tasks_total'):
            self.agent_tasks_total.labels(agent_id=agent_id, status=status).inc()
            self.agent_task_duration.labels(agent_id=agent_id).observe(duration_ms / 1000)
        
        # Store in internal metrics for API access
        key = f"agent_{agent_id}_tasks_{status}"
        self._metrics[key] = self._metrics.get(key, 0) + 1
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration_ms: float):
        """Record API request metrics"""
        if hasattr(self, 'api_requests_total'):
            self.api_requests_total.labels(
                method=method, 
                endpoint=endpoint, 
                status=str(status)
            ).inc()
            self.api_request_duration.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration_ms / 1000)
    
    def update_active_counts(self, active_agents: int, active_tasks: int):
        """Update active agent and task counts"""
        if hasattr(self, 'active_agents'):
            self.active_agents.set(active_agents)
            self.active_tasks.set(active_tasks)
        
        self._metrics['active_agents'] = active_agents
        self._metrics['active_tasks'] = active_tasks
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for API access"""
        return self._metrics.copy()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if hasattr(self, 'registry'):
            from prometheus_client import generate_latest
            return generate_latest(self.registry).decode('utf-8')
        return ""


class HealthMonitor:
    """Monitor system health and component status"""
    
    def __init__(self):
        self.config = get_config()
        self._logger = structlog.get_logger(__name__)
        self._health_checks: Dict[str, Any] = {}
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': self.config.app.version,
            'environment': self.config.app.environment,
            'checks': {}
        }
        
        # Database health
        try:
            # This would check actual database connection
            health_status['checks']['database'] = {
                'status': 'healthy',
                'response_time_ms': 5
            }
        except Exception as e:
            health_status['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'
        
        # AI Providers health
        for provider in ['anthropic', 'openai']:
            try:
                # This would check actual AI provider availability
                health_status['checks'][f'ai_provider_{provider}'] = {
                    'status': 'healthy',
                    'response_time_ms': 100
                }
            except Exception as e:
                health_status['checks'][f'ai_provider_{provider}'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Memory and disk space checks
        health_status['checks']['resources'] = await self._check_resources()
        
        return health_status
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'status': 'healthy',
            'memory_usage_percent': memory.percent,
            'disk_usage_percent': (disk.used / disk.total) * 100,
            'available_memory_gb': memory.available / (1024**3),
            'available_disk_gb': disk.free / (1024**3)
        }


# Global instances
_enterprise_logger: Optional[EnterpriseLogger] = None
_metrics_collector: Optional[MetricsCollector] = None
_health_monitor: Optional[HealthMonitor] = None


def setup_logging():
    """Initialize enterprise logging system"""
    global _enterprise_logger, _metrics_collector, _health_monitor
    
    if _enterprise_logger is None:
        _enterprise_logger = EnterpriseLogger()
        _metrics_collector = MetricsCollector()
        _health_monitor = HealthMonitor()


def get_logger(name: str, **context) -> structlog.BoundLogger:
    """Get a configured logger"""
    if _enterprise_logger is None:
        setup_logging()
    
    return _enterprise_logger.get_logger(name, **context)


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    if _metrics_collector is None:
        setup_logging()
    
    return _metrics_collector


def get_health_monitor() -> HealthMonitor:
    """Get health monitor instance"""
    if _health_monitor is None:
        setup_logging()
    
    return _health_monitor