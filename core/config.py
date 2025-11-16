"""
Configuration management for the application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "dl-postgres"
    port: int = 5432
    user: str = "admin"
    password: str = "PassW0rd"
    database: str = "db"
    connection_timeout: int = 30

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", cls.host),
            port=int(os.getenv("DB_PORT", cls.port)),
            user=os.getenv("DB_USER", cls.user),
            password=os.getenv("DB_PASSWORD", cls.password),
            database=os.getenv("DB_NAME", cls.database),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", cls.connection_timeout))
        )


@dataclass
class ModelConfig:
    """Model configuration."""
    vector_dimension: int = 2000
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    random_state: int = 42

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create config from environment variables."""
        return cls(
            vector_dimension=int(os.getenv("VECTOR_DIMENSION", cls.vector_dimension)),
            batch_size=int(os.getenv("BATCH_SIZE", cls.batch_size)),
            learning_rate=float(os.getenv("LEARNING_RATE", cls.learning_rate)),
            epochs=int(os.getenv("EPOCHS", cls.epochs)),
            early_stopping_patience=int(os.getenv("EARLY_STOPPING_PATIENCE", cls.early_stopping_patience)),
            validation_split=float(os.getenv("VALIDATION_SPLIT", cls.validation_split)),
            random_state=int(os.getenv("RANDOM_STATE", cls.random_state))
        )


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    api_url: str = "http://localhost:11434"
    embedding_model: str = "all-minilm:l6-v2"
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> 'OllamaConfig':
        """Create config from environment variables."""
        return cls(
            api_url=os.getenv("OLLAMA_API_URL", cls.api_url),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", cls.embedding_model),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", cls.max_retries))
        )


@dataclass
class FileConfig:
    """File processing configuration."""
    upload_dir: str = "/tmp/uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = None
    temp_dir: str = "/tmp/temp"

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                '.pdf', '.docx', '.doc', '.txt', '.md',
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
                '.pptx', '.xlsx', '.csv'
            ]

    @classmethod
    def from_env(cls) -> 'FileConfig':
        """Create config from environment variables."""
        return cls(
            upload_dir=os.getenv("UPLOAD_DIR", cls.upload_dir),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", cls.max_file_size)),
            temp_dir=os.getenv("TEMP_DIR", cls.temp_dir)
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "default-secret-key-change-in-production"

    database: DatabaseConfig = None
    model: ModelConfig = None
    file: FileConfig = None
    ollama: OllamaConfig = None

    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig.from_env()
        if self.model is None:
            self.model = ModelConfig.from_env()
        if self.file is None:
            self.file = FileConfig.from_env()
        if self.ollama is None:
            self.ollama = OllamaConfig.from_env()

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            secret_key=os.getenv("SECRET_KEY", cls.secret_key)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Convert nested dataclasses to dicts
        config_dict['database'] = asdict(self.database)
        config_dict['model'] = asdict(self.model)
        config_dict['file'] = asdict(self.file)
        config_dict['ollama'] = asdict(self.ollama)
        return config_dict

    def save_to_file(self, filepath: Path):
        """Save configuration to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'AppConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested configs
        db_config = DatabaseConfig(**config_dict['database'])
        model_config = ModelConfig(**config_dict['model'])
        file_config = FileConfig(**config_dict['file'])
        ollama_config = OllamaConfig(**config_dict.get('ollama', {}))

        config = cls(
            debug=config_dict['debug'],
            log_level=config_dict['log_level'],
            secret_key=config_dict['secret_key'],
            database=db_config,
            model=model_config,
            file=file_config,
            ollama=ollama_config
        )

        logger.info(f"Configuration loaded from {filepath}")
        return config


class ConfigManager:
    """Configuration manager singleton."""

    _instance: Optional[AppConfig] = None
    _config_file: Optional[Path] = None

    @classmethod
    def get_config(cls) -> AppConfig:
        """Get the current configuration instance."""
        if cls._instance is None:
            cls._instance = AppConfig.from_env()
        return cls._instance

    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> AppConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        if filepath.exists():
            cls._instance = AppConfig.load_from_file(filepath)
        else:
            logger.warning(f"Config file {filepath} not found, using environment defaults")
            cls._instance = AppConfig.from_env()

        cls._config_file = filepath
        return cls._instance

    @classmethod
    def save_config(cls, filepath: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        if filepath is None:
            filepath = cls._config_file or Path("config.json")

        cls.get_config().save_to_file(Path(filepath))


# Global config instance
config = ConfigManager.get_config()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup application logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *(logging.FileHandler(log_file) for _ in [log_file] if log_file)
        ]
    )

    logger.info(f"Logging setup with level {log_level}")


def initialize_app(config_file: Optional[str] = None):
    """Initialize the application with configuration."""
    if config_file:
        ConfigManager.load_config(config_file)

    setup_logging(config.log_level)

    logger.info("Application initialized")
    logger.info(f"Database: {config.database.host}:{config.database.port}")
    logger.info(f"Vector dimension: {config.model.vector_dimension}")
    logger.info(f"Upload directory: {config.file.upload_dir}")
    logger.info(f"Ollama API: {config.ollama.api_url} (model: {config.ollama.embedding_model})")


# Environment variable helpers
def get_env_var(name: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with validation."""
    value = os.getenv(name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable {name} is not set")

    return value


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(name: str, default: int) -> int:
    """Get integer environment variable."""
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {name}: {value}, using default {default}")
        return default


def get_env_float(name: str, default: float) -> float:
    """Get float environment variable."""
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {name}: {value}, using default {default}")
        return default