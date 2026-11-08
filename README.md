# NumberIdentifier

NumberIdentifier is a Python application that recognizes handwritten digits (0–9) from a live USB or built-in camera feed. It uses a pre-trained TensorFlow/Keras model and OpenCV to capture and prepare images for prediction.

## Features

- Captures video from the default camera.
- Predicts a digit in the centered on-screen box approximately once per second.
- Preprocesses the image by thresholding it, centering the digit, and resizing it to the model's 28 × 28 input format.
- Prints a sample prediction using the included test data when the application starts.

## Requirements

- Python 3
- A working camera, with camera access enabled for the terminal or application
- Python packages: TensorFlow, NumPy, and OpenCV

Install the packages with:

```bash
python -m pip install tensorflow numpy opencv-python
```

## Run

From the repository root, run:

```bash
python DigitReaderApp/DigitReaderApp.py
```

The application opens a camera window and displays its current prediction. Hold a handwritten digit inside the yellow box. Press **Esc** to close the window.

## Project files

- `DigitReaderApp/DigitReaderApp.py` — camera capture, image preprocessing, and prediction code.
- `DigitReaderApp/digit-recognizer.keras` — pre-trained digit recognition model.
- `DigitReaderApp/X_testreshaped.csv` and `DigitReaderApp/Y_testreshaped.csv` — test images and their labels used for the startup sample prediction.
- `DigitReaderApp.sln` — Visual Studio solution for the Python project.
