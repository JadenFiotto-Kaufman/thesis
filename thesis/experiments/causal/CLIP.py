from transformers import CLIPProcessor, CLIPModel

class CLIP:

    def __init__(self, device):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def get_inputs(self, prompts, image):

        return self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.model.device)

    def process(self, inputs):

        return self.model(**inputs)

    def __call__(self, prompts, image):

        inputs = self.get_inputs(prompts, image)

        outputs = self.process(inputs)

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)

        return logits_per_image, probs