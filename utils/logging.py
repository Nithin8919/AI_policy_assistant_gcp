"""
Logging utility - Structured logging for the AP Policy Reasoning system
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from functools import lru_cache


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, 'plan_id'):
            log_data["plan_id"] = record.plan_id
        
        if hasattr(record, 'engine'):
            log_data["engine"] = record.engine
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


@lru_cache(maxsize=1)
def setup_logger(name: str = "ap_policy_rag") -> logging.Logger:
    """
    Set up the main logger with structured formatting
    
    Args:
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler with structured JSON
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Use JSON formatter for structured logs
    # Switch to standard formatter for development if preferred
    use_json = True  # Set to False for human-readable dev logs
    
    if use_json:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "ap_policy_rag") -> logging.Logger:
    """Get or create a logger instance"""
    return logging.getLogger(name)


async def log_request(
    request_id: str,
    query: str,
    response: Any,
    trace: Dict[str, Any]
) -> None:
    """
    Log a completed request with full details
    
    This can be extended to write to BigQuery, Cloud Logging, etc.
    """
    logger = get_logger()
    
    log_entry = {
        "request_id": request_id,
        "query": query,
        "query_length": len(query),
        "answer_length": len(response.answer) if hasattr(response, 'answer') else 0,
        "num_citations": len(response.citations) if hasattr(response, 'citations') else 0,
        "used_engines": response.used_engines if hasattr(response, 'used_engines') else [],
        "confidence": response.confidence if hasattr(response, 'confidence') else 0.0,
        "processing_time_ms": response.processing_time_ms if hasattr(response, 'processing_time_ms') else 0,
        "timestamp": response.timestamp if hasattr(response, 'timestamp') else datetime.utcnow().isoformat()
    }
    
    # Add trace info
    if trace:
        log_entry["trace"] = trace
    
    logger.info(f"Request completed: {json.dumps(log_entry)}")


def log_performance_metrics(
    operation: str,
    duration_ms: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for monitoring"""
    logger = get_logger()
    
    metrics = {
        "metric_type": "performance",
        "operation": operation,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if metadata:
        metrics.update(metadata)
    
    logger.info(f"Performance: {json.dumps(metrics)}")


def log_error(
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log errors with context"""
    logger = get_logger()
    
    error_log = {
        "event_type": "error",
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if context:
        error_log["context"] = context
    
    logger.error(f"Error: {json.dumps(error_log)}")


class RequestLogger:
    """Context manager for logging request lifecycle"""
    
    def __init__(self, request_id: str, operation: str):
        self.request_id = request_id
        self.operation = operation
        self.logger = get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"Starting {self.operation}",
            extra={"request_id": self.request_id}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration:.0f}ms",
                extra={"request_id": self.request_id}
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.0f}ms: {exc_val}",
                extra={"request_id": self.request_id}
            )
        
        return False  # Don't suppress exceptions


if __name__ == "__main__":
    # Test logging
    logger = setup_logger()
    
    logger.info("System initialized")
    logger.warning("Test warning")
    logger.error("Test error")
    
    # Test structured logging
    logger.info(
        "Processing request",
        extra={
            "request_id": "test-123",
            "plan_id": "plan-456",
            "engine": "legal"
        }
    )
    
    # Test context manager
    with RequestLogger("test-request", "test_operation"):
        import time
        time.sleep(0.1)