# ANN vs CNN on MNIST (PyTorch)

Trained both an ANN and CNN on MNIST and compared them. Spoiler: CNN wins.

---

## Results

| Model | Accuracy |
|-------|----------|
| ANN   | 97.69%   |
| CNN   | 99.05%   |

CNN does better because it actually understands the 2D structure of images instead of just flattening everything like ANN does.

---

## Files

- `ann_mnist.py` — ANN model and training
- `cnn_mnist.py` — CNN model and training
- `compare.py` — plots both results together

---

## How to run

```bash
pip install torch torchvision matplotlib
python ann_mnist.py
python cnn_mnist.py
python compare.py
```

Dataset downloads automatically.

---

## Training setup

Adam optimizer, lr=0.001, batch size 32, 5 epochs. Both models use dropout and batchnorm.
