from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger("cosyvoice3_tts.normalizer")

_NORMALIZER_REGISTRY: dict[str, type[TextNormalizer]] = {}


class TextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        ...

    @abstractmethod
    def supports_language(self, lang: str) -> bool:
        ...

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            _NORMALIZER_REGISTRY[name] = subclass
            return subclass
        return wrapper


@TextNormalizer.register("identity")
class IdentityNormalizer(TextNormalizer):
    def normalize(self, text: str) -> str:
        return text

    def supports_language(self, lang: str) -> bool:
        return True


@TextNormalizer.register("ru")
class RuNormalizer(TextNormalizer):
    def __init__(self):
        self._normalizer = None
        self._available = False
        try:
            from ru_normalizr import NormalizeOptions, Normalizer
            self._normalizer = Normalizer(NormalizeOptions.tts())
            self._available = True
            _ = self._normalizer.normalize("Тест")
            logger.info("RuNormalizer ready (ru-normalizr)")
        except ImportError:
            logger.warning("ru-normalizr not installed, Russian normalization disabled")
        except Exception as e:
            logger.warning("Failed to init RuNormalizer: %s", e)

    def normalize(self, text: str) -> str:
        if not self._available or self._normalizer is None:
            return text
        if self._has_stress_markers(text):
            return text
        text = self._apply_yo_overrides(text)
        try:
            return self._normalizer.normalize(text)
        except Exception as e:
            logger.warning("ru-normalizr failed: %s", e)
            return text

    def supports_language(self, lang: str) -> bool:
        return lang == "ru" and self._available

    @staticmethod
    def _has_stress_markers(text: str) -> bool:
        return "\u0301" in text or bool(re.search(r"\+[АЕЁИОУЫЭЮЯаеёиоуыэюя]", text))

    @staticmethod
    def _apply_yo_overrides(text: str) -> str:
        overrides = {"ребенок": "ребёнок", "счет": "счёт", "счету": "счёту"}
        ru_re = re.compile(r"[А-Яа-яЁё]+", re.UNICODE)

        def replace_token(match):
            token = match.group(0)
            replacement = overrides.get(token.lower())
            if replacement is None:
                return token
            if token.isupper():
                return replacement.upper()
            if token.istitle():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        return ru_re.sub(replace_token, text)


class NormalizerPipeline:
    def __init__(self):
        self._normalizers: dict[str, list[TextNormalizer]] = {}
        self._fallback = IdentityNormalizer()

    def register(self, lang: str, normalizer: TextNormalizer) -> None:
        if lang not in self._normalizers:
            self._normalizers[lang] = []
        self._normalizers[lang].append(normalizer)
        logger.info("Registered normalizer for lang=%s: %s", lang, normalizer.__class__.__name__)

    def normalize(self, text: str, lang: str) -> str:
        normalizers = self._normalizers.get(lang, [])
        for normalizer in normalizers:
            if normalizer.supports_language(lang):
                text = normalizer.normalize(text)
        return text

    def is_available(self, lang: str) -> bool:
        return any(n.supports_language(lang) for n in self._normalizers.get(lang, []))


def create_normalizer_pipeline(enabled: bool = True) -> NormalizerPipeline:
    pipeline = NormalizerPipeline()
    if not enabled:
        return pipeline

    for name, cls in _NORMALIZER_REGISTRY.items():
        if name == "identity":
            continue
        try:
            instance = cls()
            if instance.supports_language(name):
                pipeline.register(name, instance)
            else:
                logger.info("Normalizer %s not available for lang=%s", name, name)
        except Exception as e:
            logger.warning("Failed to create normalizer %s: %s", name, e)

    return pipeline
