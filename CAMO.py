from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
from maa import EfficientNetDetector, ModelConfig  # Import your model classes

class AnimalGateDetector:
    def __init__(self, 
                 model_path: str = "final_model.pth",
                 yolo_model: str = "yolov8n.pt",
                 confidence_threshold: float = 0.3):
        """
        Initialize the detector with both YOLO and your trained model.
        
        Args:
            model_path: Path to your trained real/fake detector model
            yolo_model: Path or name of YOLO model to use
            confidence_threshold: Minimum confidence for YOLO detections
        """
        # Initialize YOLO
        self.yolo = YOLO(yolo_model)
        
        # Initialize your trained model
        self.config = ModelConfig()
        self.model = EfficientNetDetector(self.config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Define animal classes from COCO dataset that we're interested in
        self.animal_classes = {
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            # Add more animal classes as needed
        }
        
        self.confidence_threshold = confidence_threshold
        self.transform = self._get_transform()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_transform(self) -> transforms.Compose:
        """Create the transformation pipeline for images."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _contains_animals(self, results) -> Tuple[bool, List[str]]:
        """
        Check if YOLO results contain any animals.
        
        Returns:
            Tuple of (boolean indicating if animals present, list of detected animal types)
        """
        detected_animals = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                
                if (class_id in self.animal_classes and 
                    confidence >= self.confidence_threshold):
                    detected_animals.append(self.animal_classes[class_id])
        
        return len(detected_animals) > 0, list(set(detected_animals))

    def _predict_real_fake(self, image: torch.Tensor) -> Tuple[str, float]:
        """
        Predict if an image is real or synthetic.
        
        Returns:
            Tuple of (prediction string, confidence score)
        """
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            prediction = "real" if pred_class == 0 else "synthetic"
            
        return prediction, confidence

    def process_image(self, image_path: str) -> str:
        """
        Process an image through the gating mechanism.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing the detection results
        """
        # Load and process image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run YOLO detection
        results = self.yolo(image)
        
        # Check for animals
        contains_animals, detected_animals = self._contains_animals(results)
        
        if not contains_animals:
            return "No animal present in the image"
        
        # If animals are present, process through real/fake detector
        processed_image = self.transform(image)
        prediction, confidence = self._predict_real_fake(processed_image)
        
        # Format the response
        animals_found = ", ".join(detected_animals)
        return (f"Animals detected: {animals_found}\n"
                f"Image appears to be {prediction} "
                f"(confidence: {confidence:.2f})")

def main():
    # Example usage
    detector = AnimalGateDetector(
        model_path="trained_model/real_fake_detector.pth",  # Path to your trained model
        yolo_model="yolov8n.pt",      # Using YOLOv8 nano model
        confidence_threshold=0.5
    )
    
    # Test with some example images
    test_images = [
        "images/animal.jpg",
        "images/animal2.jpg",
        "images/not_animal.jpg",  # Non-animal image
    ]
    
    for image_path in test_images:
        try:
            result = detector.process_image(image_path)
            print(f"\nImage: {image_path}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()