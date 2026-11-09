# S-01: Handwritten Digit Recognition

## Risk or decision

Can the camera-based prototype recognize individual handwritten digits, and
what happens with unclear or multi-digit input?

## Set-up

The NumberIdentifier application uses a camera, OpenCV, NumPy, and a
pre-trained TensorFlow/Keras model. The camera image is preprocessed and
passed to the model for digit prediction. The application code is in
[`DigitReaderApp/DigitReaderApp.py`](../../DigitReaderApp/DigitReaderApp.py).

## How it was measured

I manually presented handwritten numbers to the camera and observed the
displayed predictions. No timing or accuracy measurements were recorded.

## Results

| Input | Observation |
| --- | --- |
| 7 | Recognized correctly. |
| 2 | Recognized correctly. |
| A very blurry, backwards 2 | The application could not read it correctly. |
| 10 | The prediction toggled between 0 and 1. |
| 3 | Recognized correctly. |

## What it confirmed

The prototype recognized 7, 2, and 3 in these manual tests, but did not
recognize the blurry, backwards 2. The model predicts individual digits
from 0 to 9, not multi-digit numbers such as 10. These observations do not
establish overall accuracy or performance.

## Related decisions and evidence

No related risk, ADR, QAS, or requirement ID was recorded. The prototype
code is available at [tag `p2`](https://github.com/drishyakoirala/Team5-SoftwareEngineering/tree/p2).
The observations above were reported from manual testing; no screenshot or
test log was recorded here.
