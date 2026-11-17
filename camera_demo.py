import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Uruchamianie demo na: {device}")

# Inicjalizacja MTCNN
mtcnn = MTCNN(keep_all=False, device=device)

# Etykiety emocji w RAF-DB
emocje = {
    0: 'Zaskoczenie (Surprise)',
    1: 'Strach (Fear)',
    2: 'Obrzydzenie (Disgust)',
    3: 'Szczecie (Happy)',
    4: 'Smutek (Sad)',
    5: 'Zlosc (Angry)',
    6: 'Neutralny (Neutral)'
}

# Sprawdzenie istnienia wytrenowanego modelu ResNet-18
MODEL_PATH = 'resnet18_rafdb.pth'
use_classifier = os.path.exists(MODEL_PATH)

if use_classifier:
    print("Znaleziono plik. Ładowanie ResNet-18...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    print("Brak pliku 'resnet18_rafdb.pth'. Uruchamiam tryb samej detekcji MTCNN.")

# 4. Uruchomienie kamery
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Detekcja ramki i 5 punktów kluczowych (oczy, nos, usta)
    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)

    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):
            if prob > 0.90:
                x1, y1, x2, y2 = [max(0, int(b)) for b in box]
                
                # Rysowanie ramki twarzy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Rysowanie 5 punktów kluczowych 
                for point in landmark:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

                label_text = f"Twarz ({prob*100:.0f}%)"

                # Klasyfikacja wyciętej twarzy
                if use_classifier:
                    face_crop = pil_img.crop((x1, y1, x2, y2))
                    if face_crop.size[0] > 0 and face_crop.size[1] > 0:
                        input_tensor = preprocess(face_crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                            conf, pred_idx = torch.max(probabilities, 0)
                            emocja_str = emocje[pred_idx.item()]
                            label_text = f"{emocja_str}: {conf.item()*100:.1f}%"

                cv2.putText(frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Demo - detekcja twarzy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()