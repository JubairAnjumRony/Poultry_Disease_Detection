# Codebase Preview & Analysis — End-to-End Deep Learning Project (Chicken Disease Classification)

---

## 1. System Overview

This is an MLOps-style deep learning project that classifies chicken fecal images into two classes: **Coccidiosis** (diseased) or **Healthy**. It uses **transfer learning with VGG16** and is deployed as a Flask web app in Docker with CI/CD to AWS (ECR + EC2).

### Directory Structure

```
.
├── app.py                          # Flask web application (serving)
├── main.py                         # Orchestrates the 4 training pipeline stages
├── config/config.yaml              # Path configurations
├── params.yaml                     # Hyperparameters
├── Dockerfile                      # Containerization
├── .github/workflows/cicd.yaml     # GitHub Actions CI/CD (ECR + self-hosted EC2)
├── requirements.txt
├── scores.json                     # Evaluation output (loss/accuracy)
├── templates/index.html            # Web UI
├── artifacts/                      # Data, base models, trained model
│   ├── data_ingestion/Chicken-fecal-images/  # Dataset (Coccidiosis, Healthy)
│   ├── prepare_base_model/         # base_model.h5, base_model_updated.h5
│   └── training/model.h5           # Final trained model
├── research/                       # Jupyter notebooks (prototyping stages)
└── src/cnnClassifier/
    ├── components/                 # Core logic per stage
    │   ├── data_ingestion.py
    │   ├── prepare_base_model.py
    │   ├── model_trainer.py
    │   └── evaluation.py
    ├── config/configuration.py     # ConfigurationManager (YAML → dataclasses)
    ├── constants/__init__.py       # CONFIG_FILE_PATH, PARAMS_FILE_PATH
    ├── entity/config_entity.py     # Frozen dataclasses for each stage config
    ├── pipeline/                   # Stage wrappers + predict.py
    └── utils/common.py             # YAML/JSON/bin IO, base64 encode/decode
```

---

## 2. How the Whole System Works

### 2.1 Training Flow (`main.py` → 4 sequential stages)

```
Stage 1: Data Ingestion        → download/extract dataset to artifacts/
Stage 2: Prepare Base Model    → load VGG16, freeze, add classifier head, save
Stage 3: Model Training        → fine-tune head on data with augmentation
Stage 4: Evaluation            → evaluate on held-out split, write scores.json
```

**Stage 1 — Data Ingestion** (`src/cnnClassifier/components/data_ingestion.py`)
- `ConfigurationManager` reads `config.yaml` and builds a `DataIngestionConfig`.
- If the dataset already exists at `artifacts/data_ingestion/Chicken-fecal-images`, it skips ingestion.
- Otherwise: if `source: local` it extracts `artifacts/data_ingestion/data.zip`; if S3 it downloads via **boto3** (with requester-pays support, region-redirect retry, atomic temp-file rename) then extracts.

**Stage 2 — Prepare Base Model** (`src/cnnClassifier/components/prepare_base_model.py`)
- Loads **VGG16** (ImageNet weights, `include_top=False`, input `224×224×3`) or reuses a cached `base_model.h5`.
- Freezes all convolutional layers, appends `Flatten → Dense(2, softmax)`.
- Compiles with `Adam(lr=1e-4)` + `CategoricalCrossentropy`, saves to `base_model_updated.h5`.

**Stage 3 — Model Training** (`src/cnnClassifier/components/model_trainer.py`)
- Loads `base_model_updated.h5`.
- Builds `ImageDataGenerator`s with `validation_split=0.20`, `rescale=1/255`.
- If `AUGMENTATION: True`: rotation(40), flip, width/height shift(0.2), shear(0.2), zoom(0.2).
- Trains for `EPOCHS=3`, `BATCH_SIZE=8` with `ModelCheckpoint` (best `val_accuracy`) → `artifacts/training/model.h5`.

**Stage 4 — Evaluation** (`src/cnnClassifier/components/evaluation.py`)
- Builds its **own** validation generator with `validation_split=0.30`, evaluates the saved model, writes `{"loss", "accuracy"}` to `scores.json` (current: **93.97% accuracy, 0.189 loss**).

### 2.2 Prediction / Serving Flow

```
Browser (index.html)
   │  POST /predict  {image: "<base64>"}
   ▼
app.py → decodeImage() writes inputImage.jpg
   ▼
PredictionPipeline.predict()
   │  load model.h5 → resize 224×224 → /255 → expand_dims
   │  model.predict → argmax → map 1="Healthy", 0="Coccidiosis"
   ▼
JSON response displayed in UI (green/red)
```

- `GET /` renders the UI; `POST /train` shells out `os.system("python main.py")`; `POST /predict` runs inference.
- Flask runs on `0.0.0.0:8080` with `debug=True`.

### 2.3 Deployment (Docker + CI/CD)

- **Dockerfile**: `python:3.8-slim-buster`, copies code, `pip install -r requirements.txt` (includes `-e .` editable install), `CMD python3 app.py`.
- **CI/CD** (`.github/workflows/cicd.yaml`): on push to `main` → integration job (placeholder lint/test) → build Docker image, push to **AWS ECR** → self-hosted runner pulls image and runs container on port 8080 with AWS env vars.

---

## 3. How CNN & Transfer Learning Is Implemented

### 3.1 The Architecture

| Component | Details |
|---|---|
| Base model | **VGG16** (ImageNet pretrained), `include_top=False` |
| Input | `(224, 224, 3)` |
| Frozen layers | All VGG16 conv blocks (14,714,688 params — 99.66% non-trainable) |
| Custom head | `Flatten()` (25088 features) → `Dense(2, activation='softmax')` |
| Trainable params | 50,178 (0.34%) |
| Optimizer | `Adam(learning_rate=0.0001)` |
| Loss | `CategoricalCrossentropy` |
| Metric | `accuracy` |

This is **feature-extraction transfer learning**: the pretrained CNN acts as a frozen feature extractor; only the final classification layer is trained.

### 3.2 Implementation (`prepare_base_model.py:30-57`)

```python
# Freeze all base layers
if freeze_all:
    for layer in model.layers:
        layer.trainable = False      # (intended; see BUG #1)

# Attach custom classifier head
flatten_in = tf.keras.layers.Flatten()(model.output)
prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)
full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

full_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)
```

### 3.3 Data Pipeline (CNN training input)

- `ImageDataGenerator` with `flow_from_directory` maps the two folder classes to one-hot labels automatically.
- Training-time augmentation (rotation, flips, shifts, shear, zoom) reduces overfitting on the small fecal-image dataset.
- Both train and inference normalize images by `1/255` — consistent preprocessing.

### 3.4 Inference (`predict.py:14-31`)

```python
test_image = image.load_img(imagename, target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image / 255.0, axis=0)
probabilities = model.predict(test_image, verbose=0)[0]
result = int(np.argmax(probabilities))
# 1 → "Healthy", 0 → "Coccidiosis"
```

---

## 4. Bugs & Critical Issues

### CRITICAL

**BUG #1 — Freezing loop sets the wrong attribute (`prepare_base_model.py:32-37`)**
```python
if freeze_all:
    for layer in model.layers:
        model.trainable = False      # ← sets MODEL flag, not LAYER flag!
elif (freeze_till is not None) and (freeze_till > 0):
    for layer in model.layers[:-freeze_till]:
        model.trainable = False      # ← same bug
```
It should be `layer.trainable = False`. As written, the loop body assigns `model.trainable` repeatedly and **no individual layer is ever frozen**. Training *appears* to work because the freshly created head is the only part with meaningful gradients at init, but in principle **all 14.7M VGG16 weights remain trainable**, which at `lr=1e-4` can rapidly destroy the pretrained features (catastrophic forgetting) and makes training far slower/more memory-hungry than intended. The model summary still reports the base as non-trainable only because the `Model` wrapper is rebuilt — but the flag is not reliably propagated. This is the most serious correctness bug in the ML core.

**BUG #2 — Evaluation uses a *different* validation split than training**
- Training: `validation_split=0.20` (`model_trainer.py:24`)
- Evaluation: `validation_split=0.30` (`evaluation.py:17`)

`flow_from_directory` splits deterministically (last N files per class = validation). Changing the ratio means the **evaluation set overlaps the training set** — some images the model was trained on are re-evaluated, inflating the reported **93.97% accuracy**. The test set is contaminated; the metric is not a true generalization estimate.

**BUG #3 — Final `save_model` overwrites the best checkpoint (`model_trainer.py:76-93`)**
`ModelCheckpoint(save_best_only=True)` saves the best-`val_accuracy` model during training, but after `fit()` completes, `self.save_model(...)` **unconditionally saves the last epoch's weights** to the same path — clobbering the best model. Whatever epoch finished last (not necessarily the best) is what ships to production.

**BUG #4 — Flask app runs with `debug=True` in production (`app.py:47`)**
```python
app.run(host='0.0.0.0', port=8080, debug=True)
```
Exposes the Werkzeug interactive debugger and PIN on a public interface → **remote code execution risk** on the deployed EC2 instance.

**BUG #5 — Unauthenticated, unvalidated `/train` endpoint shells out to the OS (`app.py:27-31`)**
```python
os.system("python main.py")
return "Training done successfully!"
```
- `os.system` uses whatever `python` is on PATH (not the venv), blocks the whole worker, ignores the exit code, and **always returns success** even if training crashed.
- Any anonymous caller can trigger expensive GPU/CPU training repeatedly (DoS / cost amplification).
- No error handling whatsoever.

**BUG #6 — Prediction loads the model from disk on *every* request (`predict.py:16`)**
`load_model()` (~15M params, several MB) runs on each `/predict` call → severe latency, memory churn, and no concurrency safety. Should load once at startup.

### HIGH

**BUG #7 — Hardcoded paths bypass the config system (`configuration.py:81-89`)**
```python
def get_validation_config(self) -> EvaluationConfig:
    eval_config = EvaluationConfig(
        path_of_model="artifacts/training/model.h5",            # hardcoded
        training_data="artifacts/data_ingestion/Chicken-fecal-images",  # hardcoded
        ...
```
Every other config comes from `config.yaml`. If paths change, training follows config but evaluation breaks. Also, these are plain `str`, not `Path` — type-inconsistent with the rest of the entities.

**BUG #8 — CWD-dependent paths in `predict.py` and `evaluation.py`**
- `predict.py:16`: `os.path.join("artifacts", "training", "model.h5")` — relative to process CWD. Works only if the app is launched from the repo root; breaks under systemd, cron, or any other working directory.
- `evaluation.py:51`: `save_json(path=Path("scores.json"), ...)` — same issue; scores land wherever CWD happens to be.

**BUG #9 — Race condition: shared fixed filename for concurrent predictions (`app.py:16, 38-40`)**
All requests write to the same `inputImage.jpg`. Two simultaneous `/predict` calls overwrite each other's images → **cross-user result leakage** (user A may get user B's diagnosis). Flask's default dev server is single-threaded, but with a production WSGI server this is a serious data-leak bug.

**BUG #10 — No input validation on `/predict` (`app.py:38`)**
- `request.json['image']` → 500 error if body isn't JSON or lacks the key (should be 400).
- No base64 validity check, no file-type check, no size limit → `decodeImage` can crash or write arbitrary/malicious content to disk (path is fixed here, but the payload isn't sanitized).

### MEDIUM

**BUG #11 — Deprecated `::set-output` in CI (`cicd.yaml:63`)**
```yaml
echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
```
GitHub deprecated & disabled `set-output`; should be `echo "image=..." >> "$GITHUB_OUTPUT"`.

**BUG #12 — CI "integration" job is a no-op (`cicd.yaml:22-26`)**
Lint and unit tests are `echo` placeholders — broken code ships straight to production. There are **no unit tests in the repo at all**.

**BUG #13 — CD deploy step commented out; redeploy will fail (`cicd.yaml:89-95`)**
The "Stop and remove container if running" step is commented out, but the next step runs `docker run --name=cnncls`. On the **second** deploy the run fails because the container name is taken — deploys work only once.

**BUG #14 — CI pins deprecated action versions (`cicd.yaml`)**
`actions/checkout@v3`, `aws-actions/configure-aws-credentials@v1`, `amazon-ecr-login@v1` are outdated (v1 credentials action lacks OIDC best practices despite `id-token: write` being declared).

**BUG #15 — Docker image bloat & fragility (`Dockerfile`)**
- `COPY . /app` copies **everything**: the full dataset under `artifacts/` (thousands of images), `.git`, notebooks, `inputImage.jpg` — image can be GBs.
- No `EXPOSE 8080`, no `.dockerignore` enforcement of artifacts (a `.dockerignore` exists but artifacts are needed at runtime for `model.h5` — conflating training data and runtime model).
- No non-root user; runs as root.
- `python:3.8-slim-buster` — Buster is EOL; Python 3.8 is EOL (Oct 2024).

**BUG #16 — `steps_per_epoch` / `validation_steps` drop the last batch (`model_trainer.py:73-74`)**
Integer division `//` truncates, so the final partial batch is silently excluded from both training and validation each epoch.

**BUG #17 — `scores.json` written to CWD, not artifacts (`evaluation.py:51`)**
Evaluation output isn't versioned with the rest of the artifacts and gets lost in deployments.

**BUG #18 — `predict.py` return schema is nonsensical**
```python
return [{"image": prediction, "prediction": prediction, "confidence": confidence}]
```
The `"image"` key contains a **class-name string**, not image data. Duplicated key, confusing API contract.

**BUG #19 — Mixed logging: `print()` vs `logger` (`data_ingestion.py:31, 49, 68, 71, 78, 84`)**
The rest of the codebase uses the package `logger`. Prints bypass log aggregation and the `logs/running_logs.log` file.

**BUG #20 — Typo: `model_evalution` (`main.py:56-57`)**
Variable name misspelled (works, but sloppy).

**BUG #21 — `get_size` imported but unused (`data_ingestion.py:7`)**; unused imports in `model_trainer.py` (`urllib.request`, `ZipFile`, `time`, `os`).

**BUG #22 — `debug=True` + `os.system` + no rate limiting = production instability**
The reloader spawns a second process; combined with `/train` this can double training runs.

**BUG #23 — `decodeImage` redundant `f.close()` inside `with` (`common.py:128-132`)** — harmless but incorrect idiom; also no try/except around `b64decode`.

**BUG #24 — No early stopping / LR reduction**
`EPOCHS: 3` with no `EarlyStopping`, no `ReduceLROnPlateau` — training is under-scheduled and blind to convergence.

**BUG #25 — Only 25,088-dim Flatten head; no regularization**
A single `Dense(2)` directly on flattened 7×7×512 VGG features with **no Dropout** and no dense hidden layer — limited capacity and high overfitting risk on a small medical-veterinary dataset. Also `Flatten` produces 25,088 params-heavy input; `GlobalAveragePooling2D` would be far more efficient.

---

## 5. Improvements

### ML / Model Quality
1. **Fix the freeze bug** — `layer.trainable = False`, then `model.trainable = False` after building the `Model` wrapper, or better: pass `trainable=False` when constructing the base model.
2. **Use a single, held-out test set** — do not derive evaluation from `validation_split`. Create an explicit `test/` directory or a fixed split manifest so the metric is honest.
3. **Save the best model correctly** — remove the unconditional `save_model()` after `fit()`, or re-load the best checkpoint before saving.
4. **Improve the head**: `GlobalAveragePooling2D → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(2, softmax)`.
5. **Fine-tune in two phases**: freeze base & train head (a few epochs), then unfreeze the last 2–3 VGG blocks and continue at `lr=1e-5`. This typically lifts accuracy substantially over head-only training.
6. **Modernize the backbone**: ResNet50V2, EfficientNetB0, or MobileNetV2 — all outperform VGG16 in accuracy/size/speed for transfer learning. VGG16's 14.8M frozen params are wasteful.
7. **Use VGG16's `preprocess_input`** (or the equivalent for the chosen backbone) instead of raw `1/255` — ImageNet-pretrained models expect their specific normalization (VGG16 uses BGR mean-subtraction).
8. **Add callbacks**: `EarlyStopping(monitor='val_loss', patience=5)`, `ReduceLROnPlateau`, `TensorBoard`.
9. **Increase epochs** (e.g. 20–50) with early stopping; 3 epochs is under-trained.
10. **Increase batch size** to 16/32 for stable gradients (8 is very noisy).
11. **Add class-weight handling** if the dataset is imbalanced.
12. **Report per-class metrics** (precision/recall/F1/confusion matrix) — accuracy alone is misleading for disease detection.

### Software Engineering
13. **Load the model once at app startup** (global/singleton in `ClientApp.__init__`) — huge latency win.
14. **Use per-request temp files or in-memory inference** (`PIL.Image.open(io.BytesIO(...))`) — eliminates the shared-file race condition.
15. **Replace `os.system` with an import or `subprocess.run(..., check=True)`** and return the real exit status; add auth + rate-limiting to `/train`, or remove it from production entirely.
16. **Turn off `debug=True`**; use gunicorn/uwsgi (`gunicorn -w 1 -b 0.0.0.0:8080 app:app`).
17. **Validate inputs** on `/predict`: check JSON keys, wrap b64decode in try/except, enforce max payload size, return proper 4xx codes.
18. **Route all paths through `config.yaml`** — remove hardcoded strings in `get_validation_config()`; use `Path` consistently.
19. **Resolve paths relative to the project root** (e.g. `Path(__file__).resolve().parents[...]`) instead of CWD.
20. **Use `logger` everywhere**; remove `print()` calls.
21. **Fix CI**: real lint (ruff/flake8) + pytest with actual tests for `common.py`, config loading, and a smoke test of the prediction pipeline.
22. **Fix CD redeploy**: re-enable the container stop/remove step; pin actions to current majors (`checkout@v4`, `configure-aws-credentials@v4`, `amazon-ecr-login@v2`); replace `::set-output` with `$GITHUB_OUTPUT`.
23. **Dockerfile hardening**: multi-stage build; copy only code + `requirements.txt` + `model.h5` (mount/download the dataset separately — training data does not belong in the serving image); add `EXPOSE 8080`; run as non-root; upgrade to `python:3.11-slim`.
24. **Add a health-check endpoint** (`/health`) for the container/ALB.
25. **Add DVC or similar** for artifact/dataset versioning (the repo structure suggests DVC-style layout but no `.dvc` files exist).
26. **Pin all dependencies** (`pyYAML`, `Flask`, `boto3` etc. are unpinned — non-reproducible builds).
27. **Move `scores.json` into `artifacts/evaluation/`** and version it.
28. **Clean up**: fix `model_evalution` typo, remove unused imports, remove redundant `f.close()` in `with` block.
29. **Add unit tests + a model-card** documenting dataset, metrics, and limitations.

### Security
30. **Never pass AWS secrets to the container via `docker run -e`** in CI logs-visible commands; use IAM roles for service accounts / instance profiles on EC2.
31. Add CORS restrictions (currently `CORS(app)` allows all origins).
32. Add authentication on `/train` and potentially `/predict`.

---

## 6. Summary

| Aspect | Status |
|---|---|
| Architecture | Clean, modular MLOps layout (components/pipelines/entities/config) |
| Transfer learning | VGG16 feature extraction + 2-unit softmax head — **freezing is buggy** |
| Reported accuracy | 93.97% — **likely inflated** due to train/eval split mismatch |
| Serving | Flask + Docker; debug mode on, model reloaded per request, shared-file race |
| CI/CD | Builds & deploys to ECR/EC2 but tests are placeholders; second deploy breaks |
| Top priority fixes | Fix `layer.trainable`, fix split contamination, fix checkpoint overwrite, disable debug, load model once |
