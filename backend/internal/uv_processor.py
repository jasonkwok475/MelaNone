import cv2
import numpy as np
from model import predict

# Expected image size for the melanoma model
IMG_SIZE = 50


def uv_map_to_image(uv_map_data):
    """
    Convert UV map data to a 2D image array.
    
    Args:
        uv_map_data: UV map data (numpy array, file path, or similar)
    
    Returns:
        numpy.ndarray: 2D grayscale image array
    """
    if isinstance(uv_map_data, str):
        # Load from file path
        uv_map = cv2.imread(uv_map_data, cv2.IMREAD_GRAYSCALE)
    elif isinstance(uv_map_data, list):
        uv_map = np.array(uv_map_data, dtype=np.uint8)
    else:
        uv_map = np.array(uv_map_data, dtype=np.uint8)
    
    # Ensure it's 2D
    if len(uv_map.shape) == 3:
        # Convert to grayscale if it's RGB
        uv_map = cv2.cvtColor(uv_map, cv2.COLOR_BGR2GRAY)
    
    return uv_map


def resize_uv_map(uv_image, target_size=IMG_SIZE):
    """
    Resize UV map image to the model's expected input size.
    
    Args:
        uv_image (numpy.ndarray): 2D image array
        target_size (int): Target size (assumes square image)
    
    Returns:
        numpy.ndarray: Resized image array
    """
    resized = cv2.resize(uv_image, (target_size, target_size))
    return resized


def normalize_image(image_array):
    """
    Normalize image array to 0-1 range.
    
    Args:
        image_array (numpy.ndarray): Image array
    
    Returns:
        numpy.ndarray: Normalized image array
    """
    normalized = image_array.astype(np.float32) / 255.0
    return normalized


def predict_from_uv_map(uv_map_data):
    """
    Complete pipeline: Convert UV map to image and predict melanoma classification.
    Returns stats formatted for frontend display.
    
    Args:
        uv_map_data: Raw UV map data (file path, numpy array, etc.)
    
    Returns:
        dict: Formatted results with keys:
            - totalObjectsAnalyzed: Number of lesions/spots analyzed (1 for single UV map)
            - concerningSpots: Number of concerning spots (melanoma predictions)
            - confidenceScores: List of confidence scores for each prediction
            - classifications: List of classifications ('melanoma' or 'benign')
    """
    try:
        # Step 1: Convert UV map to 2D image
        uv_image = uv_map_to_image(uv_map_data)
        
        # Step 2: Resize to model input size (50x50)
        uv_image_resized = resize_uv_map(uv_image, target_size=IMG_SIZE)
        
        # Step 3: Normalize
        uv_image_normalized = normalize_image(uv_image_resized)
        
        # Step 4: Add channel dimension if needed (model expects [1, 50, 50])
        if len(uv_image_normalized.shape) == 2:
            uv_image_normalized = np.expand_dims(uv_image_normalized, axis=0)
        
        # Step 5: Predict using the melanoma model
        prediction = predict(uv_image_normalized, img_size=IMG_SIZE)
        
        # Step 6: Format results for frontend
        is_melanoma = prediction['melanoma']
        melanoma_confidence = prediction['melanoma_confidence']
        benign_confidence = prediction['benign_confidence']
        
        # Count concerning spots (melanoma predictions)
        concerning_spots = 1 if is_melanoma else 0
        
        # Format for frontend
        formatted_results = {
            'totalObjectsAnalyzed': 1,
            'concerningSpots': concerning_spots,
            'confidenceScores': [melanoma_confidence, benign_confidence],
            'classifications': ['melanoma' if is_melanoma else 'benign'],
            'raw_prediction': prediction  # Include raw data for debugging
        }
        
        return formatted_results
    
    except Exception as e:
        print(f"Error in UV map prediction: {e}")
        # Return default safe values on error
        return {
            'totalObjectsAnalyzed': 1,
            'concerningSpots': 0,
            'confidenceScores': [0.5, 0.5],
            'classifications': ['unknown'],
            'error': str(e)
        }

