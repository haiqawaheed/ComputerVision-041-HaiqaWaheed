# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images
Registration No: **041** | Name: Haiqa

Nothing needs to be uploaded: the dataset downloads automatically. Run all cells from top to bottom.

```python
%pip install -q medmnist opencv-python scikit-image scikit-learn matplotlib pandas
```

## Personal Settings
Reg no **041**: seed = **41**, last digit 1 is odd so HOG cell = **16 x 16**, last digit is in 0-4 so SIFT grid = **4 x 4 (16 keypoints)**.

```python
reg_no = "041"
seed = int(reg_no[-2:])
last_digit = int(reg_no[-1])

hog_cell = 8 if last_digit % 2 == 0 else 16
grid = 4 if last_digit <= 4 else 6

print("seed =", seed)
print("HOG cell =", hog_cell)
print("SIFT grid =", grid, "x", grid)
```

```python
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from medmnist import BloodMNIST

rng = np.random.default_rng(seed)
```

# Q1. Data setup

```python
classes = {1: "eosinophil", 5: "monocyte", 6: "neutrophil", 7: "platelet"}
class_names = list(classes.values())
sizes = {"train": 100, "val": 30, "test": 30}

def get_split(split):
    data = BloodMNIST(split=split, download=True, size=64)
    images = data.imgs
    labels = data.labels.ravel()
    X, y = [], []
    for new_label, old_label in enumerate(classes):
        idx = np.where(labels == old_label)[0]
        pick = rng.choice(idx, sizes[split], replace=False)
        for i in pick:
            gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (128, 128))
            X.append(gray)
            y.append(new_label)
    return np.array(X, dtype=np.uint8), np.array(y)

X_train, y_train = get_split("train")
X_val, y_val = get_split("val")
X_test, y_test = get_split("test")

for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    print(name, "images:", X.shape, "labels:", y.shape, "class counts:", np.bincount(y).tolist())
```

```python
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for c in range(4):
    idx = np.where(y_train == c)[0][:4]
    for j in range(4):
        axes[c, j].imshow(X_train[idx[j]], cmap="gray")
        axes[c, j].axis("off")
    axes[c, 0].set_title(class_names[c], loc="left")
plt.show()
```

# Q2. HOG + SVM

```python
def get_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(hog_cell, hog_cell), cells_per_block=(2, 2))

hog_train = np.array([get_hog(img) for img in X_train])
hog_val = np.array([get_hog(img) for img in X_val])

print("HOG feature length:", hog_train.shape[1])
```

### Hand calculation (cell 16 x 16)
- Cells per side = 128 / 16 = **8**
- Blocks per side = 8 - 2 + 1 = **7**, so 7 x 7 = **49 blocks**
- Values per block = 2 x 2 cells x 9 orientations = **36**
- Total = 49 x 36 = **1764**

```python
fig, axes = plt.subplots(4, 2, figsize=(6, 12))
for c in range(4):
    i = np.where(y_train == c)[0][0]
    _, hog_picture = hog(X_train[i], orientations=9, pixels_per_cell=(hog_cell, hog_cell),
                         cells_per_block=(2, 2), visualize=True)
    axes[c, 0].imshow(X_train[i], cmap="gray")
    axes[c, 0].set_title(class_names[c])
    axes[c, 1].imshow(hog_picture, cmap="gray")
    axes[c, 1].set_title("HOG")
    axes[c, 0].axis("off")
    axes[c, 1].axis("off")
plt.show()
```

```python
best_acc = 0
for C in [0.1, 1, 10]:
    model = LinearSVC(C=C, max_iter=20000, random_state=seed)
    model.fit(hog_train, y_train)
    acc = accuracy_score(y_val, model.predict(hog_val))
    print("C =", C, "-> validation accuracy =", round(acc, 4))
    if acc > best_acc:
        best_acc = acc
        best_C = C
        hog_model = model

print("Best C:", best_C)

pred = hog_model.predict(hog_val)
cm_hog = confusion_matrix(y_val, pred)
print("Validation accuracy:", round(accuracy_score(y_val, pred), 4))
print("Macro-F1:", round(f1_score(y_val, pred, average="macro"), 4))
print(cm_hog)
ConfusionMatrixDisplay(cm_hog, display_labels=class_names).plot(xticks_rotation=30)
plt.show()
```

# Q3. SIFT + SVM

### (a) Normal SIFT detection

```python
sift = cv2.SIFT_create()

counts = []
for img in X_train:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    counts.append(len(keypoints))
counts = np.array(counts)

print("Average keypoints:", counts.mean())
print("Minimum keypoints:", counts.min())
print("Maximum keypoints:", counts.max())
print("Images with zero keypoints:", (counts == 0).sum())

keypoints, _ = sift.detectAndCompute(X_train[0], None)
drawn = cv2.drawKeypoints(X_train[0], keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(drawn)
plt.title(str(len(keypoints)) + " keypoints")
plt.axis("off")
plt.show()
```

**Why is a different number of keypoints a problem for an SVM?**
An SVM needs every image to have a feature vector of the same length, but normal SIFT gives each image a different number of descriptors.

### (b) Fixed grid of keypoints

```python
def get_grid_keypoints():
    step = 128 / grid
    keypoints = []
    for row in range(grid):
        for col in range(grid):
            x = step * col + step / 2
            y = step * row + step / 2
            keypoints.append(cv2.KeyPoint(x, y, 32))
    return keypoints

def get_sift(img):
    keypoints = get_grid_keypoints()
    keypoints, descriptors = sift.compute(img, keypoints)
    return descriptors.flatten() / 255.0

sift_train = np.array([get_sift(img) for img in X_train])
sift_val = np.array([get_sift(img) for img in X_val])

print("SIFT feature length:", sift_train.shape[1])
print("Check:", grid * grid, "x 128 =", grid * grid * 128)

drawn = cv2.drawKeypoints(X_train[0], get_grid_keypoints(), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(drawn)
plt.axis("off")
plt.show()
```

### Hand calculation
- 4 x 4 grid = **16** grid points
- Each SIFT descriptor has **128** numbers
- Total = 16 x 128 = **2048**

### (c) Train SIFT + SVM

```python
best_acc = 0
for C in [0.1, 1, 10]:
    model = LinearSVC(C=C, max_iter=20000, random_state=seed)
    model.fit(sift_train, y_train)
    acc = accuracy_score(y_val, model.predict(sift_val))
    print("C =", C, "-> validation accuracy =", round(acc, 4))
    if acc > best_acc:
        best_acc = acc
        best_C = C
        sift_model = model

print("Best C:", best_C)

pred = sift_model.predict(sift_val)
cm_sift = confusion_matrix(y_val, pred)
print("Validation accuracy:", round(accuracy_score(y_val, pred), 4))
print("Macro-F1:", round(f1_score(y_val, pred, average="macro"), 4))
print(cm_sift)
ConfusionMatrixDisplay(cm_sift, display_labels=class_names).plot(xticks_rotation=30)
plt.show()
```

# Q4. Robustness

### (a) Prediction (written before running; edit to your own opinion)
- **Rotation:** HOG will lose more. Reason: HOG cells and gradient directions are fixed, so rotating moves edges into other cells and bins.
- **Noise:** HOG will lose more. Reason: HOG uses raw pixel gradients that noise changes a lot, while SIFT smooths the image and uses a bigger 32 x 32 patch.

```python
matrix = cv2.getRotationMatrix2D((64, 64), 20, 1.0)
X_test_rot = np.array([cv2.warpAffine(img, matrix, (128, 128)) for img in X_test])

noise_rng = np.random.default_rng(seed)
noisy = X_test / 255.0 + noise_rng.normal(0, 0.05, X_test.shape)
noisy = np.clip(noisy, 0, 1)
X_test_noisy = (noisy * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, img, title in zip(axes, [X_test[0], X_test_rot[0], X_test_noisy[0]], ["clean", "rotated", "noisy"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.show()
```

### (b) Test the frozen models (the models are not trained again)

```python
results = []
for name, model, extract in [("HOG + SVM", hog_model, get_hog), ("SIFT + SVM", sift_model, get_sift)]:
    row = {"Model": name}
    for case, images in [("Clean accuracy", X_test), ("Rotated accuracy", X_test_rot), ("Noisy accuracy", X_test_noisy)]:
        features = np.array([extract(img) for img in images])
        row[case] = round(accuracy_score(y_test, model.predict(features)), 4)
    results.append(row)

table = pd.DataFrame(results)
print(table.to_string(index=False))
```

### (c) Was my prediction correct? (3 sentences, use the table above)
1. Rotation: `<correct / wrong>` because HOG went from `<x>` to `<y>` and SIFT from `<x>` to `<y>`.
2. Noise: `<correct / wrong>` because HOG went from `<x>` to `<y>` and SIFT from `<x>` to `<y>`.
3. `<one sentence reason>`

# Q5. Short answers and AI log

**(a) Why does HOG use block normalisation?**
Brightness and contrast change the size of gradients, so the same shape can give different values. Normalising each block makes the features less sensitive to these lighting changes.

**(b) Why is it wrong to choose C using the test set?**
The test set should be unseen data. If we pick C by looking at test accuracy, the test set influences the model and the accuracy we report is too optimistic.

```python
for name, cm in [("HOG", cm_hog), ("SIFT", cm_sift)]:
    best = 0
    for i in range(4):
        for j in range(i + 1, 4):
            mistakes = cm[i, j] + cm[j, i]
            if mistakes > best:
                best = mistakes
                pair = (class_names[i], class_names[j])
    print(name, "most confused:", pair, "with", best, "mistakes")
```

**(c) Most confused classes:** `<class A>` and `<class B>` (from the output above).
Grayscale makes this harder because cell types differ partly by stain colour (e.g. reddish eosinophil granules), and in grayscale these become similar grey levels.

**(d) AI log**

| Prompt (one line) | What I changed | How I checked |
|---|---|---|
| "Write notebook for HOG+SVM vs SIFT+SVM on BloodMNIST" | Set reg no 041 settings, simplified code | Printed shapes, class counts, HOG length 1764, SIFT length 2048 |
| "Add rotation and noise tests" | Applied change before feature extraction | Showed clean / rotated / noisy image |
