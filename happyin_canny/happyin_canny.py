import numpy as np
import cv2
import torch
import traceback
import sys
import math

class HappyinCanny:
    """
    HappyinCanny node for ComfyUI
    Автоматический ТРЕХпроходной Canny с ВЫСОКИМИ порогами Canny (100/200)
    для 1/2 прохода и простой фильтрацией по площади на 1м проходе.
    Без внешних настроек. Градиентная тепловая карта.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "image": ("IMAGE",), }, }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("combined_edges_final",
                    "detailed_edges_pass2",
                    "extra_detailed_edges_pass3",
                    "cleaned_first_pass_edges", # Теперь это Canny(100/200) + min_area filter
                    "detail_filter_mask",
                    "gradient_heatmap")
    FUNCTION = "apply_canny_logic"
    CATEGORY = "image/processing/Happyin"

    # Функция _filter_isolated_dust больше не нужна

    def apply_canny_logic(self, image):
        try:
            # --- Захардкоженные параметры ---
            # ---> ИЗМЕНЕНЫ ПОРОГИ CANNY для 1/2 прохода <---
            low_threshold = 100
            high_threshold = 200
            # Canny для 3го прохода (супер-детали)
            low_threshold_hot = 5
            high_threshold_hot = 20
            # Тепловая карта и маска
            blur_size = 15
            detail_threshold = 30
            hot_threshold = 150
            # ---> УПРОЩЕНА ФИЛЬТРАЦИЯ ПЕРВОГО ПРОХОДА <---
            min_area_pass1 = 5  # Убираем только самые мелкие компоненты < 5 пикселей

            # Фильтр Маски
            mask_noise_removal_size = 3
            mask_min_area = 500
            # Нечетность
            if blur_size % 2 == 0: blur_size += 1
            if mask_noise_removal_size >= 3 and mask_noise_removal_size % 2 == 0: mask_noise_removal_size += 1
            # --- Конец параметров ---

            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            batch_size, height, width, _ = image_np.shape
            results_batch = {name: [] for name in self.RETURN_NAMES}

            for b in range(batch_size):
                img_rgb_single = image_np[b]
                img_bgr_single = cv2.cvtColor(img_rgb_single, cv2.COLOR_RGB2BGR)
                original_img_filtered = cv2.GaussianBlur(img_bgr_single, (3, 3), 0)
                original_height, original_width = original_img_filtered.shape[:2]
                original_gray = cv2.cvtColor(original_img_filtered, cv2.COLOR_BGR2GRAY)

                # --- Автоматическое определение scale_factor ---
                max_dim = max(original_height, original_width)
                if max_dim > 5000: scale_factor = 1.0 / 3.0
                elif max_dim > 4000: scale_factor = 1.0 / 2.0
                elif max_dim > 1500: scale_factor = 0.7
                else: scale_factor = min(1.0, 700.0 / max_dim if max_dim > 0 else 1.0)

                # --- ПЕРВЫЙ ПРОХОД ---
                resized_width = max(1, int(original_width * scale_factor))
                resized_height = max(1, int(original_height * scale_factor))
                resized_img = cv2.resize(original_img_filtered, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                gray_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

                # Используем ВЫСОКИЕ пороги для Canny 1го прохода
                first_pass_edges_raw = cv2.Canny(gray_resized, low_threshold, high_threshold)

                # ---> УПРОЩЕННАЯ ФИЛЬТРАЦИЯ: Только по минимальной площади <---
                first_pass_edges_filtered = np.zeros_like(first_pass_edges_raw)
                if min_area_pass1 > 0 and np.count_nonzero(first_pass_edges_raw) > 0:
                    # print(f"Filtering Pass 1 by min_area={min_area_pass1}...") # DEBUG
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(first_pass_edges_raw, connectivity=8)
                    components_kept = 0
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area >= min_area_pass1:
                            first_pass_edges_filtered[labels == i] = 255
                            components_kept += 1
                    # print(f"Kept {components_kept} components after area filtering.") # DEBUG
                    # Fallback если удалили всё
                    if components_kept == 0:
                         first_pass_edges_filtered = first_pass_edges_raw
                else:
                     first_pass_edges_filtered = first_pass_edges_raw
                # ---> Конец упрощенной фильтрации <---

                # Апскейл результата первого прохода
                first_pass_edges_full = cv2.resize(first_pass_edges_filtered, (original_width, original_height),
                                                 interpolation=cv2.INTER_NEAREST)
                _, first_pass_edges_full = cv2.threshold(first_pass_edges_full, 127, 255, cv2.THRESH_BINARY)

                # --- ТЕПЛОВАЯ КАРТА И МАСКА ДЕТАЛИЗАЦИИ ---
                heat_map = cv2.GaussianBlur(first_pass_edges_full.astype(np.float32), (blur_size, blur_size), 0)
                heat_map_max = np.max(heat_map)
                if heat_map_max > 0: heat_map_normalized = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else: heat_map_normalized = np.zeros((original_height, original_width), dtype=np.uint8)

                heat_map_color = cv2.applyColorMap(heat_map_normalized, cv2.COLORMAP_JET)
                results_batch["gradient_heatmap"].append(cv2.cvtColor(heat_map_color, cv2.COLOR_BGR2RGB))

                # Фильтрация основной маски детализации
                _, high_detail_mask_raw = cv2.threshold(heat_map_normalized, detail_threshold, 255, cv2.THRESH_BINARY)
                if mask_noise_removal_size >= 3:
                    kernel_mask = np.ones((mask_noise_removal_size, mask_noise_removal_size), np.uint8)
                    high_detail_mask_morph = cv2.morphologyEx(high_detail_mask_raw, cv2.MORPH_OPEN, kernel_mask)
                else: high_detail_mask_morph = high_detail_mask_raw
                filtered_mask_final = np.zeros_like(high_detail_mask_morph)
                if mask_min_area > 0 and np.count_nonzero(high_detail_mask_morph) > 0:
                    num_labels_mask, labels_mask, stats_mask, _ = cv2.connectedComponentsWithStats(high_detail_mask_morph, connectivity=8)
                    for i in range(1, num_labels_mask):
                        if stats_mask[i, cv2.CC_STAT_AREA] >= mask_min_area:
                            filtered_mask_final[labels_mask == i] = 255
                elif np.count_nonzero(high_detail_mask_morph) > 0: filtered_mask_final = high_detail_mask_morph
                high_detail_mask_filtered = filtered_mask_final
                results_batch["detail_filter_mask"].append(cv2.cvtColor(high_detail_mask_filtered, cv2.COLOR_GRAY2RGB))

                # --- ОЧИСТКА ПЕРВОГО ПРОХОДА МАСКОЙ ---
                refined_first_pass_edges = cv2.bitwise_and(first_pass_edges_full, first_pass_edges_full, mask=high_detail_mask_filtered)
                results_batch["cleaned_first_pass_edges"].append(cv2.cvtColor(refined_first_pass_edges, cv2.COLOR_GRAY2RGB))

                # --- ВТОРОЙ ПРОХОД (стандартные детали) ---
                # Используем ВЫСОКИЕ пороги для Canny 2го прохода
                original_edges_pass2 = cv2.Canny(original_gray, low_threshold, high_threshold)
                detailed_edges_pass2 = cv2.bitwise_and(original_edges_pass2, original_edges_pass2, mask=high_detail_mask_filtered)
                results_batch["detailed_edges_pass2"].append(cv2.cvtColor(detailed_edges_pass2, cv2.COLOR_GRAY2RGB))

                # --- ТРЕТИЙ ПРОХОД (супер-детали в горячих зонах) ---
                _, very_hot_mask = cv2.threshold(heat_map_normalized, hot_threshold, 255, cv2.THRESH_BINARY)
                very_hot_mask = cv2.bitwise_and(very_hot_mask, high_detail_mask_filtered)
                # Используем НИЗКИЕ пороги для Canny 3го прохода
                original_edges_pass3 = cv2.Canny(original_gray, low_threshold_hot, high_threshold_hot)
                extra_detailed_edges_pass3 = cv2.bitwise_and(original_edges_pass3, original_edges_pass3, mask=very_hot_mask)
                results_batch["extra_detailed_edges_pass3"].append(cv2.cvtColor(extra_detailed_edges_pass3, cv2.COLOR_GRAY2RGB))

                # --- ФИНАЛЬНОЕ ОБЪЕДИНЕНИЕ ---
                combined_edges = cv2.bitwise_or(refined_first_pass_edges, detailed_edges_pass2)
                combined_edges_final = cv2.bitwise_or(combined_edges, extra_detailed_edges_pass3)
                results_batch["combined_edges_final"].append(cv2.cvtColor(combined_edges_final, cv2.COLOR_GRAY2RGB))

            # Конвертация в тензоры
            output_tensors = []
            for name in self.RETURN_NAMES:
                 tensor = torch.zeros((batch_size, height, width, 3), dtype=torch.float32).to(image.device)
                 if name in results_batch and results_batch[name]:
                     try:
                         stacked = np.stack(results_batch[name])
                         tensor = torch.from_numpy(stacked.astype(np.float32) / 255.0).to(image.device)
                     except Exception as e: print(f"ERROR converting {name}: {e}")
                 else: print(f"WARNING: List for '{name}' empty.")
                 output_tensors.append(tensor)

            if len(output_tensors) != len(self.RETURN_NAMES):
                 print(f"ERROR: Output count mismatch. Fix required.")
                 final_outputs = []
                 empty_tensor = torch.zeros((batch_size, height, width, 3), dtype=torch.float32).to(image.device)
                 for i in range(len(self.RETURN_NAMES)):
                     final_outputs.append(output_tensors[i] if i < len(output_tensors) else empty_tensor.clone())
                 output_tensors = final_outputs[:len(self.RETURN_NAMES)]

            return tuple(output_tensors)

        except Exception as e:
            print(f"ERROR in apply_canny_logic: {e}")
            traceback.print_exc()
            empty = torch.zeros_like(image); ph = empty.clone()
            names_len = len(HappyinCanny.RETURN_NAMES)
            return (ph,) * names_len