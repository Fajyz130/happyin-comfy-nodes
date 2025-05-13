import torch
import numpy as np
from PIL import Image
import ollama # Требует pip install ollama
import traceback
import math
import torch.nn.functional as F
import sys # Для sys.stdout.flush()

# --- Helper Function for Tiling ---
def simple_image_tiling(image_tensor_bhwc, tile_size=1024, overlap=64):
    if image_tensor_bhwc is None or image_tensor_bhwc.numel() == 0: return [], []
    if len(image_tensor_bhwc.shape) != 4: print("ERROR: simple_image_tiling expects BHWC input."); return [], []

    batch_size, img_h, img_w, channels = image_tensor_bhwc.shape
    # print(f"[Tiling DEBUG] Input image shape: B={batch_size}, H={img_h}, W={img_w}") # DEBUG
    if batch_size > 1: print("WARNING: simple_image_tiling processes only the first image in the batch.")

    image_chw = image_tensor_bhwc[0].permute(2, 0, 1) # HWC -> CHW
    _, img_h, img_w = image_chw.shape

    tiles = []
    coords = []
    stride = tile_size - overlap
    stride = max(1, stride) # Ensure stride is at least 1

    steps_x = math.ceil(max(0, img_w - overlap) / stride) if stride > 0 else 1
    steps_y = math.ceil(max(0, img_h - overlap) / stride) if stride > 0 else 1
    steps_x = max(1, steps_x)
    steps_y = max(1, steps_y)
    # print(f"[Tiling DEBUG] Tile grid: {steps_x}x{steps_y}, Stride: {stride}") # DEBUG

    for y_step in range(steps_y):
        for x_step in range(steps_x):
            y1 = y_step * stride
            x1 = x_step * stride
            # Adjust coordinates for the last tiles to stay within bounds
            if y1 + tile_size > img_h: y1 = max(0, img_h - tile_size)
            if x1 + tile_size > img_w: x1 = max(0, img_w - tile_size)
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            # print(f"[Tiling DEBUG] Cutting tile at y={y1}, x={x1} (size {tile_size}x{tile_size})") # DEBUG
            tile_chw = image_chw[:, y1:y2, x1:x2]
            tile_bhwc = tile_chw.permute(1, 2, 0).unsqueeze(0)
            tiles.append(tile_bhwc)
            coords.append({"x": x1, "y": y1}) # Store top-left coordinates

    # print(f"[Tiling DEBUG] Generated {len(tiles)} tiles.") # DEBUG
    return tiles, coords

# --- Основной Класс Ноды ---
class HappyinImageDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = [
            "llama3:8b-instruct", "phi3:medium-instruct", "mistral:7b-instruct",
            "mixtral:8x7b-instruct", "gemma:7b-instruct", "llava:7b", "bakllava:7b",
             "---Впишите свою модель---"
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "florence_model": ("FL2MODEL",),
                "ollama_model_name": (ollama_models, {"default": "llama3:8b-instruct"}),
                "custom_ollama_model": ("STRING", {"default": "", "multiline": False, "placeholder": "Или впишите имя модели Ollama здесь..."}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 32}),
                "max_overall_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_description", "debug_raw_descriptions")
    FUNCTION = "describe_image"
    CATEGORY = "text/processing/Happyin"

    def print_debug(self, message):
        """ Helper for printing debug messages """
        print(f"[HappyinDescriber] {message}")
        sys.stdout.flush() # Ensure immediate output

    def get_florence_description(self, image_tensor_bhwc, florence_model, task_prompt="<CAPTION>"):
        self.print_debug(f"Requesting Florence description with prompt: {task_prompt}")
        if image_tensor_bhwc is None or image_tensor_bhwc.numel() == 0:
            self.print_debug("  Input tensor is empty.")
            return ""
        if len(image_tensor_bhwc.shape) != 4:
            self.print_debug(f"  ERROR: Expected BHWC tensor, got shape {image_tensor_bhwc.shape}")
            return ""

        try:
            pil_image = Image.fromarray((image_tensor_bhwc[0].cpu().numpy() * 255).astype(np.uint8))
            self.print_debug(f"  Converted tensor to PIL Image: {pil_image.size}")

            model = florence_model['model']
            processor = florence_model['processor']
            model_dtype = florence_model.get('dtype', torch.float16)
            device = model.device
            self.print_debug(f"  Using Florence model on device: {device}, dtype: {model_dtype}")

            with torch.no_grad():
                self.print_debug("  Processing image with Florence processor...")
                inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt", do_rescale=False).to(device=device)
                # Ensure correct dtypes after moving to device
                if 'pixel_values' in inputs and inputs['pixel_values'].dtype != model_dtype:
                    inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
                if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:
                    inputs['input_ids'] = inputs['input_ids'].to(torch.long)
                self.print_debug("  Processor finished.")

                self.print_debug("  Generating Florence output...")
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                )
                self.print_debug(f"  Generated IDs shape: {generated_ids.shape}")

                results = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.print_debug("  Raw Florence result decoded.")
                # self.print_debug(f"  Raw Result: '{results[:200]}...'") # DEBUG raw output

                description = results.strip()
                # Basic post-processing
                if description.lower().startswith(task_prompt.lower()):
                    description = description[len(task_prompt):].strip()
                # print(f"  Cleaned description: '{description[:200]}...'") # DEBUG cleaned output

                return description

        except Exception as e:
            self.print_debug(f"  ERROR during Florence description: {e}")
            traceback.print_exc()
            return f"[Error describing image: {e}]"

    def synthesize_description(self, ollama_model_name, overall_desc, tile_descs):
        self.print_debug(f"Starting synthesis with Ollama model: {ollama_model_name}")
        if not ollama_model_name:
            self.print_debug("  ERROR: Ollama model name is empty.")
            return "[Error: Ollama model not specified]"

        combined_input_text = f"Общее описание сцены:\n{overall_desc}\n\n"
        combined_input_text += "Описания отдельных фрагментов:\n"
        for i, desc in enumerate(tile_descs):
            combined_input_text += f"Фрагмент {i+1}: {desc}\n"
        # self.print_debug(f"  Combined input text for Ollama:\n{combined_input_text[:500]}...") # DEBUG

        system_prompt = ("Ты - эксперт по детальному описанию изображений. Твоя задача - создать единое, связное и подробное описание изображения, "
                         "основываясь на общем обзоре и описаниях отдельных фрагментов. Интегрируй детали из описаний фрагментов в общий контекст, "
                         "устрани повторения, сгладь переходы. Не упоминай слова 'фрагмент' или номера фрагментов в финальном тексте. "
                         "Описывай изображение так, будто видишь его целиком, сохраняя максимальную детализацию.")
        user_prompt = f"ИСХОДНЫЕ ОПИСАНИЯ:\n{combined_input_text}\n\nЗАДАНИЕ: Напиши финальное, единое, детальное описание.\n\nФИНАЛЬНОЕ ОПИСАНИЕ:"

        try:
            self.print_debug("  Sending request to Ollama...")
            response = ollama.chat(
                model=ollama_model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            final_description = response['message']['content']
            self.print_debug("  Ollama synthesis successful.")
            # self.print_debug(f"  Synthesized Description: '{final_description[:200]}...'") # DEBUG result
            return final_description.strip()
        except Exception as e:
            self.print_debug(f"  ERROR interacting with Ollama model '{ollama_model_name}': {e}")
            self.print_debug("    Make sure Ollama is running and the model is pulled.")
            traceback.print_exc()
            error_message = f"[Error synthesizing: {e}]\n\n--- Raw Data ---\n{combined_input_text}"
            return error_message

    def describe_image(self, image, florence_model, ollama_model_name, custom_ollama_model,
                       tile_size=1024, tile_overlap=128, max_overall_size=1024):

        self.print_debug("--- Starting describe_image ---")
        final_ollama_model = custom_ollama_model.strip() if custom_ollama_model.strip() else ollama_model_name
        self.print_debug(f"Using Ollama model: {final_ollama_model}")
        if final_ollama_model == "---Впишите свою модель---":
             self.print_debug("  ERROR: No valid Ollama model selected or entered.")
             return ("[Error: No Ollama model specified]", "[Input image description needed]")

        # 1. Общее описание
        self.print_debug("Step 1: Generating overall description...")
        _, H, W, _ = image.shape
        self.print_debug(f"  Input image size: {W}x{H}")
        scale = min(1.0, max_overall_size / max(H, W) if max(H, W) > 0 else 1.0)
        img_resized_bhwc = image # Default if no resize needed
        if scale < 1.0:
             target_h = int(H * scale); target_w = int(W * scale)
             self.print_debug(f"  Resizing for overall description to {target_w}x{target_h} (scale: {scale:.2f})")
             try:
                 # Permute to BCHW for interpolate, then back to BHWC
                 img_resized_bhwc = F.interpolate(image.permute(0, 3, 1, 2), size=(target_h, target_w), mode='area').permute(0, 2, 3, 1)
             except Exception as e:
                 self.print_debug(f"  ERROR during overall resize: {e}. Using original image.")
                 img_resized_bhwc = image # Fallback to original if resize fails
        else:
            self.print_debug("  No resize needed for overall description.")

        overall_description = self.get_florence_description(img_resized_bhwc, florence_model, "<MORE_DETAILED_CAPTION>")
        self.print_debug(f"  Overall description generated (length: {len(overall_description)}).")

        # 2. Разбивка на тайлы
        self.print_debug(f"Step 2: Tiling image into {tile_size}x{tile_size} tiles with overlap {tile_overlap}...")
        tiles_bhwc, tile_coords = simple_image_tiling(image, tile_size, tile_overlap)
        self.print_debug(f"  Generated {len(tiles_bhwc)} tiles.")

        # 3. Описание тайлов
        tile_descriptions = []
        if not tiles_bhwc:
             self.print_debug("  WARNING: No tiles were generated.")
        else:
             self.print_debug("Step 3: Generating tile descriptions...")
             tile_device = florence_model['model'].device
             tile_dtype = florence_model.get('dtype', torch.float16)
             for i, tile in enumerate(tiles_bhwc):
                 self.print_debug(f"  Describing tile {i+1}/{len(tiles_bhwc)} (Coords: x={tile_coords[i]['x']}, y={tile_coords[i]['y']})")
                 tile_desc = self.get_florence_description(tile.to(device=tile_device, dtype=tile_dtype), florence_model, "<CAPTION>")
                 tile_descriptions.append(tile_desc)
                 # self.print_debug(f"    Desc: '{tile_desc[:80]}...'") # DEBUG tile desc

        # 4. Синтез описания через Ollama
        self.print_debug("Step 4: Synthesizing final description...")
        final_description = self.synthesize_description(final_ollama_model, overall_description, tile_descriptions)
        self.print_debug(f"  Final description length: {len(final_description)}")

        # 5. Возвращаем результат
        self.print_debug("Step 5: Preparing outputs...")
        raw_descriptions_for_debug = f"Overall:\n{overall_description}\n\nTiles ({len(tile_descriptions)}):\n" + "\n".join([f"{i+1}: {d}" for i, d in enumerate(tile_descriptions)])
        self.print_debug("--- describe_image finished ---")

        return (final_description, raw_descriptions_for_debug)