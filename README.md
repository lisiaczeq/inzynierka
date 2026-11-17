# Rozpoznawanie emocji z twarzy (FER)

Projekt z zakresu wizji komputerowej realizujący detekcję twarzy oraz klasyfikację emocji w czasie rzeczywistym z wykorzystaniem biblioteki **PyTorch**, detektora **MTCNN** oraz sieci **ResNet-18** trenowanej na zbiorze **RAF-DB**.

## Funkcjonalności

- **Detekcja twarzy i punktów charakterystycznych:** Wykorzystanie modelu MTCNN (`facenet-pytorch`) do wyznaczania ramki ograniczającej (bounding box) oraz 5 kluczowych punktów twarzy (oczy, nos, kąciki ust).
- **Klasyfikacja 7 emocji podstawowych:** Model ResNet-18 (transfer learning z wag ImageNet) klasyfikujący wycięty obszar twarzy do jednej z 7 klas ze zbioru RAF-DB:
  1. Zaskoczenie (*Surprise*)
  2. Strach (*Fear*)
  3. Obrzydzenie (*Disgust*)
  4. Szczęście (*Happy*)
  5. Smutek (*Sad*)
  6. Złość (*Angry*)
  7. Neutralny (*Neutral*)
- **Demo w czasie rzeczywistym:** Obsługa strumienia wideo z kamery internetowej za pomocą OpenCV (z automatycznym trybem awaryjnym samej detekcji MTCNN, gdy brak pliku z wagami modelu).

## Struktura projektu

```text
.
├── archive/
│   └── DATASET/            # Zbiór danych RAF-DB (ignorowany przez Git)
│       ├── train/          # Foldery klas treningowych (1..7)
│       └── test/           # Foldery klas testowych (1..7)
├── camera_demo.py          # Skrypt uruchamiający podgląd na żywo z kamery
├── train_baseline.py       # Skrypt trenujący model bazowy ResNet-18
├── resnet18_rafdb.pth      # Zapisane wagi wytrenowanego modelu (ignorowane przez Git)
└── .gitignore
```

## Wymagania i instalacja

1. Utwórz i aktywuj środowisko wirtualne:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Zainstaluj wymagane pakiety:
   ```powershell
   pip install torch torchvision facenet-pytorch opencv-python Pillow
   ```
   *(Dla przyspieszenia obliczeń na karcie graficznej NVIDIA zalecana jest instalacja wersji PyTorch ze wsparciem CUDA ze strony [pytorch.org](https://pytorch.org/)).*

## Przygotowanie danych

Umieść zbiór danych **RAF-DB** (w formacie kompatybilnym z `torchvision.datasets.ImageFolder`) w ścieżce:
- `./archive/DATASET/train/`
- `./archive/DATASET/test/`

## Uruchomienie

### 1. Trening modelu bazowego
Aby przeprowadzić trening sieci ResNet-18 na zbiorze RAF-DB i zapisać wagi do pliku `resnet18_rafdb.pth`:
```powershell
python train_baseline.py
```

### 2. Demo z kamerą internetową
Aby uruchomić detekcję twarzy i rozpoznawanie emocji na żywo z kamery:
```powershell
python camera_demo.py
```
- Naciśnij klawisz **`q`** w oknie podglądu, aby zakończyć działanie programu.
