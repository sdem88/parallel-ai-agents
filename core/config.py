"""
Enterprise configuration management with validation
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog


logger = structlog.get_logger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    url: str = Field(default="sqlite:///./agents_system.db")
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=0, ge=0)
    echo: bool = Field(default=False)
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class SecurityConfig(BaseSettings):
    """Security configuration"""
    jwt_secret_key: str = Field(min_length=32)
    api_key: str = Field(min_length=16)
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24, ge=1, le=168)
    
    # CORS settings
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string
            return [origin.strip() for origin in v.split(",")]
        return v


class AIProviderConfig(BaseSettings):
    """AI Provider configuration"""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Default models
    anthropic_model: str = Field(default="claude-3-opus-20240229")
    openai_model: str = Field(default="gpt-4")
    
    # Provider settings
    max_tokens: int = Field(default=4096, ge=100, le=100000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=10, le=300)
    
    # Retry configuration
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    
    model_config = SettingsConfigDict(env_prefix="AI_")


class AgentConfig(BaseSettings):
    """Agent system configuration"""
    max_concurrent_agents: int = Field(default=10, ge=1, le=100)
    task_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    health_check_interval_seconds: int = Field(default=60, ge=10, le=600)
    
    # Agent-specific settings
    max_tasks_per_agent: int = Field(default=5, ge=1, le=50)
    agent_idle_timeout_minutes: int = Field(default=30, ge=5, le=480)
    
    model_config = SettingsConfigDict(env_prefix="AGENT_")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO")
    structured: bool = Field(default=True)
    file_path: Optional[str] = None
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=50)
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    @validator("level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration"""
    sentry_dsn: Optional[str] = None
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1000, le=65535)
    
    # Health check settings
    health_check_enabled: bool = Field(default=True)
    health_check_path: str = Field(default="/health")
    
    # Metrics collection
    collect_detailed_metrics: bool = Field(default=True)
    metrics_retention_hours: int = Field(default=168, ge=1, le=8760)  # 1 week default
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class ApplicationConfig(BaseSettings):
    """Main application configuration"""
    debug: bool = Field(default=False)
    testing: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1000, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    
    # Application metadata
    name: str = Field(default="Agent System")
    version: str = Field(default="2.0.0")
    description: str = Field(default="Enterprise AI Agent System")
    
    model_config = SettingsConfigDict(env_prefix="APP_")
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()


class EnterpriseConfig:
    """Main configuration class that combines all settings"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration from environment variables"""
        self._env_file = env_file or ".env"
        
        # Load all configuration sections
        self.app = ApplicationConfig(_env_file=self._env_file)
        self.database = DatabaseConfig(_env_file=self._env_file)
        self.security = SecurityConfig(_env_file=self._env_file)
        self.ai = AIProviderConfig(_env_file=self._env_file)
        self.agents = AgentConfig(_env_file=self._env_file)
        self.logging = LoggingConfig(_env_file=self._env_file)
        self.monitoring = MonitoringConfig(_env_file=self._env_file)
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(
            "Configuration loaded successfully",
            environment=self.app.environment,
            debug=self.app.debug
        )
    
    def _validate_configuration(self):
        """Perform cross-configuration validation"""
        validations = []
        
        # Production environment validations
        if self.app.environment == "production":
            if self.app.debug:
                validations.append("Debug mode should not be enabled in production")
            
            if not self.security.jwt_secret_key or len(self.security.jwt_secret_key) < 32:
                validations.append("JWT secret key must be at least 32 characters in production")
            
            if "*" in self.security.cors_origins:
                validations.append("CORS should not allow all origins in production")
            
            if not self.monitoring.sentry_dsn:
                validations.append("Sentry DSN should be configured in production")
        
        # AI Provider validations
        if not self.ai.anthropic_api_key and not self.ai.openai_api_key:
            validations.append("At least one AI provider API key must be configured")
        
        # Resource validations
        if self.agents.max_concurrent_agents * self.agents.max_tasks_per_agent > 500:
            validations.append("Total concurrent tasks might be too high for system resources")
        
        if validations:
            for validation in validations:
                logger.warning("Configuration validation warning", issue=validation)
            
            if self.app.environment == "production":
                raise ValueError(f"Production configuration issues: {'; '.join(validations)}")
    
    def get_ai_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific AI provider"""
        base_config = {
            "max_tokens": self.ai.max_tokens,
            "temperature": self.ai.temperature,
            "timeout": self.ai.timeout_seconds
        }
        
        if provider.lower() == "anthropic":
            return {
                **base_config,
                "api_key": self.ai.anthropic_api_key,
                "model": self.ai.anthropic_model
            }
        elif provider.lower() == "openai":
            return {
                **base_config,
                "api_key": self.ai.openai_api_key,
                "model": self.ai.openai_model
            }
        else:
            raise ValueError(f"Unknown AI provider: {provider}")
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.app.environment == "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (safe for logging)"""
        return {
            "app": self.app.dict(exclude={"debug"}),
            "database": self.database.dict(exclude={"url"}),
            "security": {"cors_origins": self.security.cors_origins},
            "ai": {"providers_configured": [
                "anthropic" if self.ai.anthropic_api_key else None,
                "openai" if self.ai.openai_api_key else None
            ]},
            "agents": self.agents.dict(),
            "logging": self.logging.dict(),
            "monitoring": self.monitoring.dict(exclude={"sentry_dsn"})
        }


# Global configuration instance
config: Optional[EnterpriseConfig] = None


def get_config() -> EnterpriseConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config = EnterpriseConfig()
    return config


def reload_config(env_file: Optional[str] = None) -> EnterpriseConfig:
    """Reload configuration (useful for testing)"""
    global config
    config = EnterpriseConfig(env_file)
    return config