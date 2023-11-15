import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientvit.seg_model_zoo import create_seg_model
from efficientvit.models.utils import resize

def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(image)
        output = torch.argmax(output.squeeze(), dim=0)
    return output.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the Cityscapes image")
    parser.add_argument("--model", type=str, default="l2", help="Model name")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the model weight file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_seg_model(args.model, "cityscapes", weight_url=None)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model = model.to(device).eval()

    segmentation = predict(args.image_path, model, device)
    # Do something with the segmentation result, e.g., save it or display it

    print(segmentation)

if __name__ == "__main__":
    main()
