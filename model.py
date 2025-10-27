"""
COMPLETE POTHOLE DETECTION TRAINING & INFERENCE PIPELINE
For images with NO annotations or bounding boxes

Directory Structure Required:
dataset/
├── train/
│   ├── pothole/        # Images WITH potholes
│   └── no_pothole/     # Images WITHOUT potholes
└── val/
    ├── pothole/
    └── no_pothole/

For YOLO training (optional - requires annotation):
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import shutil


# ==================== STEP 1: DATA PREPARATION ====================

class PotholeDatasetBuilder:
    """Build dataset from raw images"""
    
    @staticmethod
    def organize_images(source_dir, output_dir, val_split=0.2):
        """
        Organize images into train/val split
        
        Args:
            source_dir: Directory with subdirectories 'pothole' and 'no_pothole'
            output_dir: Output directory for organized dataset
            val_split: Validation split ratio
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        for category in ['pothole', 'no_pothole']:
            category_path = source_path / category
            if not category_path.exists():
                print(f"Warning: {category_path} not found")
                continue
            
            # Get all images
            images = list(category_path.glob('*.jpg')) + \
                    list(category_path.glob('*.png')) + \
                    list(category_path.glob('*.jpeg'))
            
            # Split into train and val
            train_imgs, val_imgs = train_test_split(
                images, test_size=val_split, random_state=42
            )
            
            # Copy to organized structure
            for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
                dest_dir = output_path / split / category
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for img in img_list:
                    shutil.copy(img, dest_dir / img.name)
            
            print(f"{category}: {len(train_imgs)} train, {len(val_imgs)} val")
    
    @staticmethod
    def augment_images(image_dir, augment_factor=3):
        """
        Create augmented versions of images
        """
        image_path = Path(image_dir)
        images = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        
        augmentations = [
            lambda img: cv2.flip(img, 1),  # Horizontal flip
            lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=10),  # Brightness
            lambda img: cv2.convertScaleAbs(img, alpha=0.8, beta=-10),  # Darkness
        ]
        
        for img_path in tqdm(images, desc="Augmenting"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            for i in range(min(augment_factor, len(augmentations))):
                aug_img = augmentations[i](img.copy())
                new_name = img_path.stem + f"_aug{i}" + img_path.suffix
                cv2.imwrite(str(img_path.parent / new_name), aug_img)


# ==================== STEP 2: CLASSIFICATION MODEL TRAINING ====================

class PotholeImageDataset(Dataset):
    """Dataset for pothole classification"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load pothole images (label = 1)
        pothole_dir = self.root_dir / 'pothole'
        if pothole_dir.exists():
            for img_path in pothole_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        # Load no_pothole images (label = 0)
        no_pothole_dir = self.root_dir / 'no_pothole'
        if no_pothole_dir.exists():
            for img_path in no_pothole_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    self.images.append(str(img_path))
                    self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ClassifierTrainer:
    """Train pothole classifier"""
    
    def __init__(self, train_dir, val_dir, model_save_path='pothole_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model_save_path = model_save_path
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        self.train_dataset = PotholeImageDataset(train_dir, self.train_transform)
        self.val_dataset = PotholeImageDataset(val_dir, self.val_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        # Create model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 2)
        )
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {'train_loss': [], 'train_acc': [], 
                       'val_loss': [], 'val_acc': []}
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 
                            'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, epochs=20):
        print(f"\nStarting training for {epochs} epochs...")
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"Model saved! Best Val Acc: {best_val_acc:.2f}%")
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
        self.plot_history()
        
        return self.model
    
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("Training history saved to training_history.png")


# ==================== STEP 3: ANNOTATION TOOL FOR YOLO ====================

class SimpleAnnotationTool:
    """Simple tool to create bounding box annotations for YOLO"""
    
    def __init__(self, image_dir, output_dir):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.images = list(self.image_dir.glob('*.jpg')) + \
                     list(self.image_dir.glob('*.png'))
        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            self.boxes.append((self.start_point, end_point))
            
    def save_annotations(self, img_path, img_shape):
        """Save in YOLO format"""
        h, w = img_shape[:2]
        label_path = self.output_dir / (img_path.stem + '.txt')
        
        with open(label_path, 'w') as f:
            for (x1, y1), (x2, y2) in self.boxes:
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h
                
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
    
    def annotate(self):
        """Interactive annotation"""
        print("\nAnnotation Tool Instructions:")
        print("- Click and drag to draw bounding boxes")
        print("- Press 's' to save and next image")
        print("- Press 'd' to delete last box")
        print("- Press 'q' to quit")
        
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        while self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            img = cv2.imread(str(img_path))
            display_img = img.copy()
            
            # Draw existing boxes
            for (x1, y1), (x2, y2) in self.boxes:
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw current box being drawn
            if self.drawing and self.start_point:
                cv2.circle(display_img, self.start_point, 5, (0, 0, 255), -1)
            
            # Show info
            cv2.putText(display_img, f"Image {self.current_idx+1}/{len(self.images)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_img, f"Boxes: {len(self.boxes)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Annotation Tool', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_annotations(img_path, img.shape)
                print(f"Saved {len(self.boxes)} boxes for {img_path.name}")
                self.boxes = []
                self.current_idx += 1
            elif key == ord('d'):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()


# ==================== STEP 4: YOLO TRAINING (if annotations available) ====================

class YOLOTrainer:
    """Train YOLOv8 for pothole detection"""
    
    def __init__(self, data_yaml_path):
        self.data_yaml = data_yaml_path
        self.model = YOLO('yolov8n.pt')  # Start with pretrained weights
    
    def train(self, epochs=100, img_size=640, batch_size=16):
        """Train YOLO model"""
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='pothole_detection',
            patience=20,
            save=True,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        return results
    
    @staticmethod
    def create_data_yaml(dataset_path, output_path='pothole_data.yaml'):
        """Create YOLO data.yaml file"""
        yaml_content = f"""
path: {dataset_path}
train: images/train
val: images/val

nc: 1
names: ['pothole']
"""
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created {output_path}")
        return output_path


# ==================== STEP 5: COMPLETE INFERENCE SYSTEM ====================

class PotholeClassifier:
    """Binary classifier to check if pothole exists"""
    
    def __init__(self, model_path, conf_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 2)
        )
        
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded classifier from {model_path}")
        else:
            raise FileNotFoundError(f"Classifier model not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, frame):
        with torch.no_grad():
            img = self.transform(frame).unsqueeze(0).to(self.device)
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            has_pothole = predicted.item() == 1
            conf_value = confidence.item()
            
            return has_pothole and conf_value > self.conf_threshold, conf_value


class PotholeDetectorTracker:
    """YOLO-based detector and tracker"""
    
    def __init__(self, yolo_model_path, conf_threshold=0.5):
        if not Path(yolo_model_path).exists():
            print(f"Warning: YOLO model not found at {yolo_model_path}")
            print("Using pre-trained YOLOv8n. Train your own model for better results.")
            yolo_model_path = 'yolov8n.pt'
        
        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        
    def detect_and_track(self, frame):
        results = self.model.track(
            frame, 
            persist=True,
            conf=self.conf_threshold,
            verbose=False
        )
        return results
    
    def draw_detections(self, frame, results):
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [None] * len(boxes)
            
            for box, conf, track_id in zip(boxes, confidences, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                label = f'Pothole {conf:.2f}'
                if track_id is not None:
                    label = f'ID:{track_id} {label}'
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        return annotated_frame


class PotholeMultiModelSystem:
    """Complete inference system"""
    
    def __init__(self, classifier_path, yolo_path, 
                 classification_conf=0.7, detection_conf=0.5):
        self.classifier = PotholeClassifier(classifier_path, classification_conf)
        self.detector = PotholeDetectorTracker(yolo_path, detection_conf)
        self.fps = 0
        self.classification_enabled = True
        
    def process_frame(self, frame):
        start_time = time.time()
        
        has_pothole = False
        class_confidence = 0.0
        
        if self.classification_enabled:
            has_pothole, class_confidence = self.classifier.predict(frame)
        else:
            has_pothole = True
        
        detection_results = None
        annotated_frame = frame.copy()
        
        if has_pothole:
            detection_results = self.detector.detect_and_track(frame)
            annotated_frame = self.detector.draw_detections(frame, detection_results)
            status_text = f"Pothole Detected! (Conf: {class_confidence:.2f})"
            cv2.putText(annotated_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            status_text = f"No Pothole (Conf: {class_confidence:.2f})"
            cv2.putText(annotated_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        self.fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, has_pothole, detection_results
    
    def process_camera(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        print("Camera started... Press 'q' to quit, 'c' to toggle classification")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, has_pothole, _ = self.process_frame(frame)
            cv2.imshow('Pothole Detection & Tracking', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.classification_enabled = not self.classification_enabled
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, _, _ = self.process_frame(frame)
            
            if writer:
                writer.write(annotated_frame)
            
            cv2.imshow('Processing Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


# ==================== MAIN WORKFLOW ====================

if __name__ == "__main__":
    
    # ========== WORKFLOW SELECTION ==========
    
    print("=" * 60)
    print("POTHOLE DETECTION TRAINING & INFERENCE PIPELINE")
    print("=" * 60)
    print("\nSelect workflow:")
    print("1. Organize and prepare dataset")
    print("2. Train classification model")
    print("3. Annotate images for YOLO (interactive)")
    print("4. Train YOLO model")
    print("5. Run inference (image/video/camera)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    # ========== 1. ORGANIZE DATASET ==========
    if choice == '1':
        print("\n--- Dataset Organization ---")
        source_dir = input("Enter source directory path (with pothole/ and no_pothole/ folders): ").strip()
        output_dir = input("Enter output directory path: ").strip() or "dataset"
        
        PotholeDatasetBuilder.organize_images(source_dir, output_dir)
        
        augment = input("\nAugment training data? (y/n): ").strip().lower()
        if augment == 'y':
            for category in ['pothole', 'no_pothole']:
                img_dir = Path(output_dir) / 'train' / category
                if img_dir.exists():
                    PotholeDatasetBuilder.augment_images(img_dir, augment_factor=3)
        
        print("\n✓ Dataset prepared successfully!")
    
    # ========== 2. TRAIN CLASSIFIER ==========
    elif choice == '2':
        print("\n--- Training Classification Model ---")
        train_dir = input("Enter training directory path: ").strip() or "dataset/train"
        val_dir = input("Enter validation directory path: ").strip() or "dataset/val"
        epochs = int(input("Enter number of epochs (default 20): ").strip() or "20")
        
        trainer = ClassifierTrainer(train_dir, val_dir)
        trainer.train(epochs=epochs)
        
        print("\n✓ Classifier trained successfully!")
        print(f"Model saved to: pothole_classifier.pth")
    
    # ========== 3. ANNOTATE FOR YOLO ==========
    elif choice == '3':
        print("\n--- Interactive Annotation Tool ---")
        image_dir = input("Enter directory with pothole images: ").strip()
        output_dir = input("Enter output directory for labels: ").strip() or "yolo_labels"
        
        annotator = SimpleAnnotationTool(image_dir, output_dir)
        annotator.annotate()
        
        print("\n✓ Annotation completed!")
    
    # ========== 4. TRAIN YOLO ==========
    elif choice == '4':
        print("\n--- Training YOLO Model ---")
        dataset_path = input("Enter YOLO dataset path: ").strip()
        
        # Create data.yaml
        yaml_path = YOLOTrainer.create_data_yaml(dataset_path)
        
        epochs = int(input("Enter number of epochs (default 100): ").strip() or "100")
        
        yolo_trainer = YOLOTrainer(yaml_path)
        yolo_trainer.train(epochs=epochs)
        
        print("\n✓ YOLO model trained successfully!")
        print("Check runs/detect/pothole_detection/weights/best.pt")
    
    # ========== 5. RUN INFERENCE ==========
    elif choice == '5':
        print("\n--- Running Inference ---")
        classifier_path = input("Enter classifier model path: ").strip() or "pothole_classifier.pth"
        yolo_path = input("Enter YOLO model path: ").strip() or "yolov8n.pt"
        
        print("\nSelect input source:")
        print("1. Camera (live)")
        print("2. Video file")
        print("3. Image file")
        
        source_choice = input("Enter choice (1-3): ").strip()
        
        system = PotholeMultiModelSystem(
            classifier_path=classifier_path,
            yolo_path=yolo_path,
            classification_conf=0.7,
            detection_conf=0.5
        )
        
        if source_choice == '1':
            camera_id = int(input("Enter camera ID (default 0): ").strip() or "0")
            system.process_camera(camera_id)
        
        elif source_choice == '2':
            video_path = input("Enter video file path: ").strip()
            output_path = input("Enter output path (optional): ").strip() or None
            system.process_video(video_path, output_path)
        
        elif source_choice == '3':
            image_path = input("Enter image file path: ").strip()
            frame = cv2.imread(image_path)
            if frame is not None:
                annotated, has_pothole, _ = system.process_frame(frame)
                cv2.imshow('Result', annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                save = input("Save result? (y/n): ").strip().lower()
                if save == 'y':
                    output_path = input("Enter output path: ").strip() or "output.jpg"
                    cv2.imwrite(output_path, annotated)
                    print(f"Saved to {output_path}")
        
        print("\n✓ Inference completed!")
    
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)