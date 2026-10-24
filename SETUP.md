# PPR501 Car Analysis - Setup

This repository contains three independently run applications:

| Source | Application | Start command |
| --- | --- | --- |
| `sources/CNN` | CNN image classifier (MobileNetV2) with a Gradio UI | `python .\sources\CNN\train_cnn.py --mode serve` |
| `sources/LinearReg` | Linear Regression plot and Streamlit UI | See [Linear Regression](#4-run-the-linear-regression-app) |
| `sources/Transformer` | Transformer-based car recommendation chatbot | See [Transformer chatbot](#5-run-the-transformer-recommendation-chatbot) |

All Python libraries for the three apps are listed in the single root
[`requirements.txt`](./requirements.txt). Create one virtual environment at the
repository root and install dependencies from that file.

Full documentation for the CNN app lives in
[`sources/CNN/README.md`](./sources/CNN/README.md).

## 1. Prerequisites and installation

The commands below are for Windows PowerShell. Install Python 3.10 or 3.11 and
ensure the Python Launcher (`py`) is available.

Run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
python -m ipykernel install --user --name ppr501-car-analysis --display-name "Python (PPR501 Car Analysis)"
```

Activate the same `.venv` in each terminal used to run an app. If PowerShell
blocks environment activation, invoke the environment's Python directly, for
example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## 2. Check the datasets

The data should remain in the following locations:

```text
sources/
└── Dataset/
    ├── car-price.csv
    ├── Train/
    │   ├── Audi/
    │   └── ...
    └── Test/
        ├── Audi/
        └── ...
```

`car-price.csv` is used by both `LinearReg` and `Transformer`. `Train` and
`Test` contain the seven car-image class folders used by `CNN`. Keep the
folder names and structure intact.

## 3. Run the CNN image-classification app

Full documentation (architecture, metrics, troubleshooting) is in
[`sources/CNN/README.md`](./sources/CNN/README.md).

`train_cnn.py` is the single source of truth; `main.ipynb` is a thin wrapper
around it, so both entry points behave identically.

From the repository root, with the shared `.venv` activated:

```bash
cd sources/CNN

# Quick sanity check (a few batches, about 15 seconds)
python train_cnn.py --mode train --smoke --epochs 2 --freeze-epochs 1

# Full training + evaluation (MobileNetV2, 20 epochs)
python train_cnn.py --mode train --epochs 20 --freeze-epochs 5

# Evaluate the saved model again
python train_cnn.py --mode evaluate

# Predict a single image
python train_cnn.py --mode predict --image ../Dataset/Test/Audi/1000.jpg

# Launch the Gradio image-upload interface
python train_cnn.py --mode serve
```

The Gradio interface opens at `http://localhost:7860`. Stop it with `Ctrl+C`.
Add `--share` to create a temporary public URL (do not upload private images
while that setting is on).

Training artifacts are written to `sources/CNN/outputs/car_cnn/`
(`model.keras`, `labels.json`, `meta.json`, `history.json`, `metrics.json`,
`confusion_matrix.png`, `training_curves.png`).

### Notebook

`main.ipynb` wraps the same script, so the results are identical:

```powershell
jupyter lab .\sources\CNN\main.ipynb
```

In JupyterLab, select the **Python (PPR501 Car Analysis)** kernel, then run the
notebook cells in order (or choose **Run All**). Training runs for 20 epochs
and may take a while. The final cell contains the Gradio interface; uncomment
`demo.launch(...)` to start it.

> On macOS, if Matplotlib cannot write its cache, run
> `export MPLCONFIGDIR=/tmp/mplcache` before training.

## 4. Run the Linear Regression app

### Streamlit interface

From the repository root, with the shared `.venv` activated:

```powershell
streamlit run .\sources\LinearReg\app.py --server.port 8501
```

Open `http://localhost:8501`. Select an input feature and target, then choose
**Train Linear Regression** to display the evaluation metrics, regression plot,
and prediction control.

### Standalone regression plot

The standalone script expects its working directory to be `sources\LinearReg`.
In a separate PowerShell terminal opened at the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\sources\LinearReg
python .\linear_regression.py
```

Close the Matplotlib window to finish the script.

## 5. Run the Transformer recommendation chatbot

### Streamlit interface

From the repository root, with the shared `.venv` activated:

```powershell
streamlit run .\sources\Transformer\app.py --server.port 8502
```

Open `http://localhost:8502` and enter a question about cars in the dataset.
The first run downloads the `paraphrase-multilingual-MiniLM-L12-v2` model from
Hugging Face, so internet access is required; later runs can reuse the cached
model.

### Jupyter notebook

The notebook provides the same dataset filtering and Transformer ranking flow
as `app.py`, displaying the top matches in a table. From the repository root,
with the shared `.venv` activated:

```powershell
Set-Location .\sources\Transformer
jupyter lab .\main.ipynb
```

Select the **Python (PPR501 Car Analysis)** kernel, run the cells in order, and
enter a car request in the final cell. The notebook also downloads the model
from Hugging Face the first time it runs.

## Running the apps together

Each app runs as its own process. Start it in a separate terminal and leave
that terminal open. The default ports in the commands above are:

* Linear Regression Streamlit: `8501`
* Transformer chatbot Streamlit: `8502`
* CNN Gradio: the local URL/port printed by Gradio (normally `7860`)

Stop each app with `Ctrl+C` in its terminal, or stop the active notebook cell
for CNN or Transformer.

## Troubleshooting

* **A module is missing:** activate the root `.venv` and reinstall with
  `python -m pip install -r .\requirements.txt`.
* **Dataset or CSV file is not found:** verify the data locations shown above.
  Run `linear_regression.py` from `sources\LinearReg` as instructed.
* **The CNN kernel cannot import TensorFlow:** select the **Python (PPR501 Car
  Analysis)** kernel so Jupyter uses the root `.venv`.
* **Transformer startup is slow:** its pretrained embedding model is downloaded
  on first use; allow the download to finish and ensure internet access.
