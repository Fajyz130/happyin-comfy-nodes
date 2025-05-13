# __init__.py для расширения Happyin ComfyUI

# Импорт двух нод, которые не требуют дополнительных зависимостей
from .happyin_canny import HappyinCanny
from .happyin_word_replacer import HappyinWordReplacer

# Словари для экспорта узлов
NODE_CLASS_MAPPINGS = {
    "HappyinCanny": HappyinCanny,
    "HappyinWordReplacer": HappyinWordReplacer,
}

# Отображаемые имена узлов
NODE_DISPLAY_NAME_MAPPINGS = {
    "HappyinCanny": "Happyin Canny",
    "HappyinWordReplacer": "Happyin Word Replacer",
}

# Проверяем доступность ollama для третьей ноды - HappyinImageDescriber
try:
    import ollama
    from .happyin_image_describer import HappyinImageDescriber
    # Добавляем третью ноду, если ollama доступен
    NODE_CLASS_MAPPINGS["HappyinImageDescriber"] = HappyinImageDescriber
    NODE_DISPLAY_NAME_MAPPINGS["HappyinImageDescriber"] = "Happyin Image Describer"
    print("Happyin Image Describer доступен!")
except ImportError:
    print("Внимание: пакет 'ollama' не найден. HappyinImageDescriber не будет доступен.")
    print("Для использования Image Describer установите ollama: pip install ollama")

print("Happyin ComfyUI расширение успешно загружено с", len(NODE_CLASS_MAPPINGS), "нодами!")