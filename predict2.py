import argparse
import torch
import os
import glob
import pandas as pd
import time
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
    city = 'Wellington'
    continent = 'Oceania'
    default_folder_path = f'/home/tur50665/Seasonal SVI/data/{continent}/{city}/GSV'
    default_weight_path = 'assets/checkpoints/seg/cityscapes/l2.pt'

    default_pid_file = city + '_pid.csv'

    parser.add_argument("--folder_path", type=str, default=default_folder_path,
                        help="Path to the folder containing the test_dir")
    parser.add_argument("--model", type=str, default="l2", help="Model name")
    parser.add_argument("--weight_path", type=str, default=default_weight_path, help="Path to the model weight file")
    parser.add_argument("--pid_file", type=str, default=default_pid_file, help="Path to the PID file")
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = create_seg_model(args.model, "cityscapes", weight_url=None)
    model.load_state_dict(torch.load(args.weight_path, map_location=device)['state_dict'])
    model = model.to(device).eval()

    result_file = f'seg_results/{city}_segmentation_results1.csv'
    # result_id_file = 'segmentation_results_id.csv'

    class_labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                    'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    # 检查 result_file 是否存在，如果存在，则读取已处理的 image_name
    processed_images = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for line in f:
                image_name = line.split(',')[0]
                processed_images.add(image_name)

    if not os.path.exists(result_file):
        with open(result_file, 'w', encoding='utf-8') as f:
            header = 'img_id,' + ','.join(class_labels) + '\n'
            f.write(header)

    n = 0

    image_paths = glob.glob(os.path.join(args.folder_path, '*.jpg'))

    # Process each image in the folder
    for image_path in image_paths:
        image_name = os.path.basename(image_path).split('.')[0]
        if image_name in processed_images:
            continue  # 如果图片已经处理过，跳过
        n += 1
        segmentation = predict(image_path, model, device)
        total_pixels = segmentation.size
        class_counts = np.bincount(segmentation.ravel(), minlength=19)
        class_proportions = class_counts / total_pixels

        with open(result_file, 'a', encoding='utf-8') as f:
            line = f'{image_name},' + ','.join(map(str, class_proportions)) + '\n'
            f.write(line)

        print(f'---------Image {image_name} the {n}th segmentation calculation completed--------')
    # Read PID file and segmentation results, then merge
    # pid_data = pd.read_csv(args.pid_file)
    # segmentation_results = pd.read_csv(result_file)
    # merged_data = pd.merge(pid_data, segmentation_results, left_on='pid', right_on='img_id')
    # merged_data.drop(columns=['img_id']).to_csv(result_id_file, index=False)


if __name__ == "__main__":
    main()
