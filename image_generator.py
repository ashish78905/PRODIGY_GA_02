

from transformers import DALLMiniForConditionalGeneration, DALLMiniProcessor
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Load the pre-trained DALL-E Mini model and processor
dalle_model_name = "dalle-mini/dalle-mini"
dalle_model = DALLMiniForConditionalGeneration.from_pretrained(dalle_model_name).to("cuda")
dalle_processor = DALLMiniProcessor.from_pretrained(dalle_model_name)

def generate_dalle_image(prompt):
    inputs = dalle_processor(prompt, return_tensors="pt").to("cuda")
    outputs = dalle_model.generate(inputs.input_ids, max_length=512)
    image = Image.fromarray(outputs.cpu().numpy().astype('uint8'))
    return image

# Load the pre-trained Stable Diffusion model
stable_model_id = "CompVis/stable-diffusion-v1-4"
stable_pipe = StableDiffusionPipeline.from_pretrained(stable_model_id).to("cuda")

def generate_stable_image(prompt):
    image = stable_pipe(prompt).images[0]
    return image

# Generate images based on prompts
dalle_prompt = "A fantasy landscape with castles and dragons"
stable_prompt = "A serene beach at sunset"

dalle_image = generate_dalle_image(dalle_prompt)
stable_image = generate_stable_image(stable_prompt)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(dalle_image)
plt.title('DALL-E Mini')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stable_image)
plt.title('Stable Diffusion')
plt.axis('off')

plt.show()
