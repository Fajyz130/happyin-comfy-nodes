import re
import torch # Для совместимости с ComfyUI

class HappyinWordReplacer:
    """
    HappyinWordReplacer node for ComfyUI
    Заменяет "негативные" слова (пыль, царапины, дефекты и т.д.)
    в тексте на более "позитивные" или нейтральные аналоги.
    Версия без слова "shiny".
    """

    # Словарь замен: "негатив" -> "позитив/нейтрал"
    REPLACEMENT_DICT = {
        # --- Пыль, Грязь, Пятна, Посторонние частицы ---
        "dust": "clean",
        "dusty": "clean",
        "dirt": "clean",
        "dirty": "clean",
        "grime": "clean",
        "grimy": "clean",
        "lint": "clean",
        "hair": "clean",
        "hairs": "clean",
        "fiber": "clean",
        "fibers": "clean",
        "speck": "spotless",
        "specks": "spotless",
        "particle": "spotless surface",
        "particles": "spotless surface",
        "debris": "clean surface",
        "residue": "clean",
        "contaminant": "clean",
        "contamination": "clean",

        # --- Пятна, Отпечатки, Следы ---
        "stain": "spotless",
        "stains": "spotless",
        "stained": "spotless",
        "smudge": "spotless",
        "smudges": "spotless",
        "smudged": "spotless",
        "fingerprint": "spotless",
        "fingerprints": "spotless",
        "grease": "clean",
        "oily": "clean",
        "smear": "spotless",
        "smears": "spotless",
        "watermark": "spotless",
        "watermarks": "spotless",
        "waterspot": "spotless",
        "waterspots": "spotless",
        "mark": "spotless",
        "marks": "spotless",
        "spot": "spotless",
        "spots": "spotless",
        "blotch": "spotless",
        "blotches": "spotless",
        "splotch": "spotless",
        "splotches": "spotless",
        "blemish": "flawless",
        "blemishes": "flawless",

        # --- Царапины, Потертости, Сколы, Вмятины ---
        "scratch": "smooth finish",
        "scratches": "smooth finish",
        "scratched": "smooth finish",
        "micro-scratch": "flawless finish",
        "hairline": "flawless finish",
        "scuff": "polished finish", # Используем polished finish
        "scuffs": "polished finish",
        "scuffed": "polished finish",
        "abrasion": "smooth texture",
        "abrasions": "smooth texture",
        "gouge": "smooth surface",
        "scrape": "smooth surface",
        "scrapes": "smooth surface",
        "scraped": "smooth surface",
        "scoring": "smooth surface",
        "score": "smooth surface",
        "swirl": "mirror finish",
        "swirls": "mirror finish",
        "etching": "smooth surface", # Гравюра/травление -> гладкая поверхность
        "chip": "perfect edge",
        "chips": "perfect edges",
        "chipped": "perfect edge",
        "dent": "smooth contour",
        "dents": "smooth contour",
        "dented": "smooth contour",
        "pit": "smooth surface",
        "pits": "smooth surface",
        "pitted": "smooth surface",
        "pitmark": "smooth surface",

        # --- Трещины, Разломы, Структурные дефекты ---
        "crack": "seamless",
        "cracks": "seamless",
        "cracked": "seamless",
        "fracture": "intact",
        "fractures": "intact",
        "fractured": "intact",
        "break": "intact",
        "broken": "intact",
        "tear": "intact edge",
        "torn": "intact edge",

        # --- Общие Дефекты, Несовершенства ---
        "damage": "pristine condition",
        "damaged": "pristine condition",
        "defect": "perfect design",
        "defective": "perfect design",
        "flaw": "flawless",
        "flawed": "flawless",
        "imperfection": "perfect",
        "imperfections": "perfect",
        "imperfect": "perfect",
        "burr": "smooth edge",
        "burrs": "smooth edge",
        "ragged": "smooth edge",
        "jagged": "smooth edge",

        # --- Износ ---
        "wear": "new condition", # "New condition" вместо "pristine"
        "wearing": "new condition",
        "worn": "new condition",
        "bruise": "unmarked",

        # --- Прозрачность, Ясность ---
        "haze": "crystal clear",
        "hazy": "crystal clear",
        "cloudiness": "crystal clear",
        "cloudy": "crystal clear",
        "film": "clear surface", # Уточнено
        "veil": "clear surface",
        "fog": "clear",
        "foggy": "clear",
        "milkiness": "clear",
        "milky": "clear",
        "opaque": "transparent",
        "translucent": "transparent",

        # --- Цвет, Блеск, Окисление ---
        "tarnish": "polished finish", # Используем polished finish
        "tarnished": "polished finish",
        "discoloration": "uniform color",
        "discolorations": "uniform color",
        "faded": "vibrant color",
        "oxide": "polished finish",
        "oxidation": "polished finish",
        "oxidised": "polished finish",
        "oxidised-layer": "polished finish",
        "corrosion": "polished finish", # Тоже к полировке
        "rust": "polished finish",
        "rusty": "polished finish",
        "dull": "bright surface", # Используем bright surface
        "dullness": "bright surface",
        "lusterless": "lustrous", # Используем lustrous (лоснящийся, глянцевый)
        "matte": "smooth matte finish", # ИЗМЕНЕНО: сохраняем матовость, но делаем ее идеальной
        "matting": "smooth matte finish",
        # "patina": "bright finish", # ЗАКОММЕНТИРОВАНО: патина слишком специфична

        # --- Текстура, Неровность ---
        "roughness": "smoothness",
        "rough": "smooth",
        "uneven": "even surface",
        "irregular": "symmetrical",
        "streaky": "uniform texture",
        "blotchy": "uniform color",

        # --- Качество, Разрешение ---
        "bad": "excellent",
        "poor": "excellent",
        "ugly": "elegant", # Заменено с unique
        "unpleasant": "pleasant",
        "cheap": "luxurious",
        "flimsy": "sturdy",
        "low quality": "high quality",
        "low-quality": "high-quality",
        "low res": "high resolution",
        "low-res": "high-resolution",

        # --- Фокус, Размытие ---
        "blur": "sharp focus",
        "blurry": "sharp focus",
        "blurred": "sharp focus",
        # "out of focus": "sharp focus",
        # "unfocused": "sharp focus",

        # --- Шум, Зерно ---
        "noise": "smooth texture", # Оставим текстуру, но гладкую
        "noisy": "smooth texture",
        "grain": "fine texture", # Оставим текстуру, но мелкую
        "grainy": "fine texture",
    }

    # Предварительная компиляция паттернов для возможного ускорения (делается один раз при загрузке класса)
    # Сортируем ключи по длине (от длинных к коротким) для правильной замены фраз
    COMPILED_PATTERNS = []
    for negative_word in sorted(REPLACEMENT_DICT.keys(), key=len, reverse=True):
         # Компилируем регулярное выражение для каждого слова
         pattern = re.compile(r'\b' + re.escape(negative_word) + r'\b', flags=re.IGNORECASE)
         replacement = REPLACEMENT_DICT[negative_word]
         COMPILED_PATTERNS.append((pattern, replacement))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_text",)
    FUNCTION = "replace_words"
    CATEGORY = "text/processing/Happyin" # Та же категория, что и Canny

    def replace_words(self, text):
        modified_text = text
        # Итерируемся по предкомпилированным паттернам
        for pattern, replacement in self.COMPILED_PATTERNS:
            modified_text = pattern.sub(replacement, modified_text)

        return (modified_text,)

# Убедитесь, что этот код сохранен как happyin_word_replacer.py
# и обновлен __init__.py, как в предыдущем ответе.