import os
import sys
import torch

# Add parent directories to Python path to find melanoma_classifier
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from melanoma_classifier.melanoma_image_classifier.net_class import Net

MODEL_PATH = os.path.join(ROOT_DIR, 'melanoma_classifier/melanoma_image_classifier/saved_model.pth')

def load_model():
    model = Net()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
    model.eval()
    return model

def predict(image_array, img_size = 100):
    """
    Predict whether the given image is melanoma or benign.

    Args:
        image_array (numpy.ndarray): 3D image array of shape img_size (default 100).
    Returns:
        dict: Prediction results with keys 'melanoma', 'benign_confidence', 'melanoma_confidence'.
    """
    model = load_model()

    img_array = image_array / 255.0  # Normalize
    img_tensor = torch.Tensor(img_array)

    with torch.no_grad():
        output = model(img_tensor.view(-1, 1, img_size, img_size))[0]

    is_melanoma = output[1].item() >= output[0].item()

    return {
        'melanoma': is_melanoma,
        'benign_confidence': round(float(output[0]), 3),
        'melanoma_confidence': round(float(output[1]), 3)
    }