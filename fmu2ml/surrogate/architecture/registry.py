"""
Architecture registry — extension + factory pattern.

Allows users to register custom architectures alongside the built-in
phases and retrieve them by name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Built-in factories are imported lazily to avoid circular imports.


class SurrogateRegistry:
    """Singleton registry mapping architecture names to factory callables."""

    _instance: Optional["SurrogateRegistry"] = None
    _registry: Dict[str, Callable] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._register_builtins()
        return cls._instance

    def _register_builtins(self):
        self._registry["lstm"] = "fmu2ml.surrogate.architecture.lstm:lstm"
        self._registry["deeponet"] = "fmu2ml.surrogate.architecture.deeponet:deeponet"
        self._registry["hybrid_deeponet"] = (
            "fmu2ml.surrogate.architecture.hybrid_deeponet:hybrid_deeponet"
        )
        self._registry["domain_deeponet"] = (
            "fmu2ml.surrogate.architecture.domain_deeponet:domain_deeponet"
        )
        self._registry["federated"] = (
            "fmu2ml.surrogate.architecture.federated:federated"
        )
        self._registry["federated_pi"] = (
            "fmu2ml.surrogate.architecture.federated_pi:federated_pi"
        )

    def register(self, name: str, factory: Callable):
        """Register a custom architecture factory.

        Parameters
        ----------
        name : str
            Short name, e.g. ``"my_transformer"``.
        factory : Callable
            ``factory(config, **kwargs) -> nn.Module``.
        """
        self._registry[name] = factory

    def get(self, name: str) -> Callable:
        """Return factory callable for *name*; resolve lazy strings."""
        entry = self._registry.get(name)
        if entry is None:
            raise KeyError(
                f"Unknown architecture '{name}'. "
                f"Available: {list(self._registry.keys())}"
            )
        # Lazy import from dotted path
        if isinstance(entry, str):
            module_path, func_name = entry.rsplit(":", 1)
            import importlib

            mod = importlib.import_module(module_path)
            factory = getattr(mod, func_name)
            self._registry[name] = factory
            return factory
        return entry

    def list(self) -> List[str]:
        return list(self._registry.keys())


def list_architectures() -> List[str]:
    """Return names of all registered architectures."""
    return SurrogateRegistry().list()