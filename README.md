# Multilingual Image Generation with Text Prompt

This project demonstrates the generation of images based on textual prompts using a stable diffusion model. The text prompts are translated into multiple languages using Google Translate before generating images.

## Installation

To install the necessary packages, run the following commands:

```bash
!pip install googletrans==3.1.0a0
!pip install --upgrade diffusers transformers -q
```

## Usage

### Define the configuration parameters for the image generation process.
1. Set up the Configuration
```bash
from googletrans import Translator
from diffusers import StableDiffusionPipeline
import torch

class CFG:
  device = "cuda"
  seed = 42
  generator = torch.Generator(device).manual_seed(seed)
  image_gen_steps = 35
  image_gen_model_id = "stabilityai/stable-diffusion-2"
  image_gen_size = (900,900)
  image_gen_guidance_scale = 9
  prompt_gen_model_id = "gpt3"
  prompt_dataset_size = 6
  prompt_max_length = 12
```

### Initialize the stable diffusion model for image generation.
2. Initialize the Image Generation Model
```bash
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, 
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN",
    guidance_scale=9
)

image_gen_model = image_gen_model.to(CFG.device)
```

### Define a function to translate text into the desired language using Google Translate.
3. Define Translation Function
```bash
def get_translation(text, dest_lang):
  translator = Translator()
  translated_text = translator.translate(text, dest_lang)
  return translated_text.text
```

4. Generate Image
```bash
def generate_image(prompt, model):
  translation = get_translation(prompt, "en")
  image = model(
      translation, num_inference_steps=CFG.image_gen_steps,
      generator=CFG.generator,
      guidance_scale=CFG.image_gen_guidance_scale 
  ).images[0]

  image = image.resize(CFG.image_gen_size)
  return image
```
### Provide an example of how to use the functions to generate an image based on a text prompt.
5. Example Usage
```bash
prompt = "മരത്തിൽ ഇരിക്കുന്നു ഒരു മത്സ്യം"
image = generate_image(prompt, image_gen_model)
image.show()
```

## License
This project is licensed under the MIT License.
