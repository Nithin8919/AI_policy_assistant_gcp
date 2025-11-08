"""
Configuration management for AP Policy Reasoning System

Loads and validates settings from YAML
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """
    Load configuration from settings.yaml
    
    Cached to avoid repeated file I/O
    """
    config_path = Path(__file__).parent / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that all required configuration fields are present"""
    required_sections = ["models", "project", "engines", "routing"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate engines
    if not config["engines"]:
        raise ValueError("No RAG engines configured")
    
    for name, engine in config["engines"].items():
        if "id" not in engine:
            raise ValueError(f"Engine '{name}' missing 'id' field")
        if "weight" not in engine:
            raise ValueError(f"Engine '{name}' missing 'weight' field")
    
    # Validate project settings
    if config["project"]["gcp_project_id"] == "YOUR_GCP_PROJECT_ID":
        raise ValueError(
            "Please update project.gcp_project_id in config/settings.yaml with your GCP project ID"
        )


def get_engine_config(engine_name: str) -> Dict[str, Any]:
    """Get configuration for a specific engine"""
    config = load_config()
    
    if engine_name not in config["engines"]:
        raise ValueError(f"Unknown engine: {engine_name}")
    
    return config["engines"][engine_name]


def get_all_facets() -> Dict[str, list]:
    """Get mapping of engines to their facets"""
    config = load_config()
    return {
        name: engine.get("facets", [])
        for name, engine in config["engines"].items()
    }


if __name__ == "__main__":
    # Test configuration loading
    try:
        cfg = load_config()
        print("✅ Configuration loaded successfully")
        print(f"Engines configured: {list(cfg['engines'].keys())}")
        print(f"LLM model: {cfg['models']['llm']}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

