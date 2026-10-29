# Windows PC Setup and Demonstration

This guide covers a local Windows 10/11 setup, starting the Streamlit app, and
demonstrating both video tracking and reference-image search across the sample
videos.

## Requirements

- 64-bit Windows 10 or 11
- Python 3.12 (64-bit) and Git
- At least 8 GB RAM; 16 GB recommended
- Internet access for the first model download
- NVIDIA GPU is optional; the CPU setup works without CUDA

## 1. Get the project

Install Git and Python 3.12 if they are not already installed. During Python
installation, enable **Add python.exe to PATH**.

Open PowerShell and clone the repository:

```powershell
git clone https://github.com/prxsnn/person-reid-multicamera.git
cd person-reid-multicamera
```

If you already have the project, open PowerShell in its
`person-reid-multicamera` folder instead.

Check that Python 3.12 is available:

```powershell
py -3.12 --version
```

## 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for this PowerShell window only,
then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The prompt should start with `(venv)`. Alternatively, use Command Prompt:

```bat
venv\Scripts\activate.bat
```

## 3. Download the model weights

The app uses two pretrained model files:

- **YOLO11m detector:** [Download `yolo11m.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)
  and save it in the project root, alongside `app.py`.
- **OSNet-x0.25 ReID model:** [Download the MSMT17 checkpoint](https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth)
  and save it as `weights\osnet_x0_25_msmt17.pth`.

Alternatively, download both files from PowerShell in the project root:

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt" `
  -OutFile "yolo11m.pt"
Invoke-WebRequest `
  -Uri "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth" `
  -OutFile "weights\osnet_x0_25_msmt17.pth"
```

Both files are large model weights and are excluded from Git. The app can
download the weights automatically when they are missing, but downloading them
manually is useful if automatic download is blocked.

## 4. Install dependencies

For the simplest setup, use CPU-only PyTorch:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Use the project virtual environment whenever you install packages or launch the
app. The `venv` folder is local to your PC and is intentionally not committed to
Git; `requirements.txt` describes the Python dependencies.

## 5. Start the app

From the project folder, with `(venv)` active:

```powershell
python -m streamlit run app.py
```

Open <http://localhost:8501> in your browser. The first run may download the
YOLO and OSNet model weights; allow it to finish. Model weights are local runtime
files and are intentionally excluded from Git.

## 6. Demonstrate video tracking

1. Select the **Track an uploaded video** tab.
2. Upload `test_data/videos/two_persons_cross.mp4`.
3. Keep the default sidebar settings and click **Start Tracking**.
4. Review the annotated video and tracking logs, then try **Download CSV Logs**.
5. To show another scenario, repeat with `test_data/videos/reentry_gap.mp4` or
   `test_data/videos/crowd_five.mp4`.

The colored boxes and `ID:N` labels show confirmed tracks. This demo tracks
people within the selected video; it does not search for an uploaded image.

## 7. Demonstrate reference-image search across all test videos

1. Select the **Find a person in test videos** tab.
2. Upload `test_data/images/query/query_person_A.png` as the reference image.
   You can also use a clear image containing one visible person.
3. Leave **Maximum cosine distance** at its default initially.
4. Click **Search all test videos** and wait for the progress indicator to finish.
5. Review the result table. It reports each searched video, matching time
   ranges, number of matching frames, similarity, and detector confidence.
6. Expand a match to see the best matching frame with the detected person
   outlined. Use **Download search results** to save the CSV.
7. Try `test_data/images/query/query_person_B.png` or change the distance
   threshold. A lower maximum distance is stricter; a higher value may return
   more matches, including false positives.

The search scans every supported video under `test_data/` (including nested
folders). It uses appearance similarity, not facial recognition. Results are
best-effort: different clothing, pose, lighting, image quality, and camera view
can cause missed or incorrect matches. The supplied videos and query images are
synthetic demo assets, not a guarantee of matching real people.

## Troubleshooting

- **`py -3.12` is not recognized:** install 64-bit Python 3.12 and enable the
  Python launcher/PATH option.
- **Activation is blocked:** use the process-scoped PowerShell command above or
  activate from Command Prompt.
- **Port 8501 is busy:** stop the other Streamlit process, or run
  `python -m streamlit run app.py --server.port 8502` and open
  <http://localhost:8502>.
- **Model download fails:** check internet access and rerun the app. Do not put
  downloaded model files in a Git commit.
- **Search finds no matches:** use a clearer full-body/person crop and adjust
  the cosine-distance threshold gradually. Appearance search is not guaranteed
  to identify every person.

## Environment variables and secrets

The current app does not read project-specific environment variables or a
`.env` file. Its tracking defaults are in `tracker/config.py`; Python package
versions are in `requirements.txt`. Keep the virtual environment, model weights,
and any future credentials out of Git. If the project later needs environment
variables, document their names and safe placeholder values in a committed
`.env.example`, and keep actual secret values in a local ignored `.env` file or
your deployment provider's secret settings.
