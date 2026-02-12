"""Plugin registries for extensible components."""

from typing import Callable, Dict, Any

STT_PLUGINS: Dict[str, Callable[..., Any]] = {}
EMOTION_PLUGINS: Dict[str, Callable[..., Any]] = {}
LLM_PLUGINS: Dict[str, Callable[..., Any]] = {}


def register_stt(name: str, factory: Callable[..., Any]):
    STT_PLUGINS[name] = factory


def register_emotion(name: str, factory: Callable[..., Any]):
    EMOTION_PLUGINS[name] = factory


def register_llm(name: str, factory: Callable[..., Any]):
    LLM_PLUGINS[name] = factory


def get_stt(name: str):
    return STT_PLUGINS.get(name)


def get_emotion(name: str):
    return EMOTION_PLUGINS.get(name)


def get_llm(name: str):
    return LLM_PLUGINS.get(name)
