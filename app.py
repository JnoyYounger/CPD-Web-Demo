import os
import uuid
from flask import Flask, request, jsonify, send_file, render_template
import torch
import numpy as np
from PIL import Image
import cv2
from model.CPD_models import CPD_VGG
from torchvision import transforms
import metrics as mt

app = Flask(__name__)

UPLOAD_FOLDER = 'Uploads'
RESULT_FOLDER = 'Results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CPD_VGG().to(device)
model_path = os.path.join(os.path.dirname(__file__), 'CPD.pth')
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功加载模型权重：{model_path}")
except Exception as e:
    print(f"加载模型权重失败：{e}")
model.eval()

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)[0]
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    print(f"预测值范围：min={pred.min()}, max={pred.max()}")
    pred = (pred * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred)
    unique_id = uuid.uuid4().hex
    pred_path = os.path.join(RESULT_FOLDER, f'pred_mask_{unique_id}.png')
    pred_image.save(pred_path)
    print(f"预测 mask 保存至：{pred_path}")
    pred_image = pred_image.resize(image.size, Image.BILINEAR)
    pred_array = np.array(pred_image)
    mask = cv2.threshold(pred_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print(f"Mask 非零像素比例：{np.count_nonzero(mask) / mask.size:.4f}")
    if np.count_nonzero(mask) / mask.size > 0.99:
        print("警告：Mask 几乎全白，可能模型预测失败")
        return None, None
    original = np.array(image)
    foreground = cv2.bitwise_and(original, original, mask=mask)
    foreground_pil = Image.fromarray(foreground)
    foreground_path = os.path.join(RESULT_FOLDER, f'foreground_only_{unique_id}.png')
    foreground_pil.save(foreground_path)
    print(f"前景保存至：{foreground_path}")
    white_background = np.ones_like(original) * 255
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_background, white_background, mask=mask_inv)
    background_pil = Image.fromarray(background)
    background_path = os.path.join(RESULT_FOLDER, f'background_only_{unique_id}.png')
    background_pil.save(background_path)
    print(f"背景保存至：{background_path}")
    final_image = cv2.add(foreground, background)
    foreground_pil = Image.fromarray(final_image)
    result_path = os.path.join(RESULT_FOLDER, f'foreground_{unique_id}.png')
    foreground_pil.save(result_path)
    return result_path, mask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("错误：未上传文件")
        return {'error': '未上传文件'}, 400
    file = request.files['file']
    if file.filename == '':
        print("错误：未选择文件")
        return {'error': '未选择文件'}, 400
    unique_id = uuid.uuid4().hex
    filename = f'uploaded_{unique_id}.{file.filename.rsplit(".", 1)[1].lower()}'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print(f"原图保存至：{file_path}")
    print(f"处理图像：{file_path}")
    result_path, mask = process_image(file_path)
    if result_path is None:
        print("错误：抠图失败")
        return {'error': '抠图失败，可能模型预测无效'}, 500
    print(f"抠图保存至：{result_path}")
    file_path_url = file_path.replace(os.sep, '/')
    result_path_url = result_path.replace(os.sep, '/')
    response = {
        'original': f'/{file_path_url}',
        'foreground': f'/{result_path_url}',
        'metrics': {}
    }
    if 'gt_file' in request.files and request.files['gt_file'].filename != '':
        gt_file = request.files['gt_file']
        gt_unique_id = uuid.uuid4().hex
        gt_filename = f'gt_{gt_unique_id}.{gt_file.filename.rsplit(".", 1)[1].lower()}'
        gt_path = os.path.join(UPLOAD_FOLDER, gt_filename)
        gt_file.save(gt_path)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_image is not None and mask is not None:
            metrics_dict = mt.compute_metrics(mask, gt_image)
            response['metrics'] = {
                'mae': float(metrics_dict['mae']),
                'fmeasure': float(metrics_dict['fmeasure']),
                'precision': float(metrics_dict['precision']),
                'recall': float(metrics_dict['recall']),
                'iou': float(metrics_dict['iou'])
            }
            print(f"指标：{response['metrics']}")
        else:
            print("警告：无法加载真实标注图像或 mask 无效")
    print(f"返回响应：{response}")
    return jsonify(response)

@app.route('/Uploads/<path:filename>')
def serve_uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"请求原图：{file_path}")
    return send_file(file_path)

@app.route('/Results/<path:filename>')
def serve_result_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    print(f"请求抠图：{file_path}")
    return send_file(file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))