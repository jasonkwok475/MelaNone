import os
import torch

from melanoma_classifier.melanoma_image_classifier.net_class import Net

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../melanoma_classifier/melanoma_image_classifier/saved_model.pth')

def load_model():
    model = Net()
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model.load_state_dict(torch.load(model_path, weights_only=True))
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