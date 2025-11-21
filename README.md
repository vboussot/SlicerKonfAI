# SlicerKonfAI

SlicerKonfAI is a 3D Slicer module that provides a graphical user interface (GUI) for running **KonfAI Apps** directly inside 3D Slicer.

It is designed as the “clinical front-end” for KonfAI: you train and package your models as KonfAI Apps, and SlicerKonfAI lets clinicians load volumes, select an app, run inference, and visualize the outputs in Slicer. It also provides built-in Quality Assurance (QA) tools, including reference-based metrics and reference-free uncertainty estimation.

---

## ✨ Features

### **• Hugging Face–Hosted Apps**
SlicerKonfAI automatically discovers and downloads *KonfAI Apps* hosted on Hugging Face. No manual installation is required: select an App from the list and it becomes instantly available. You can also add your own Apps to the workspace.

### **• Fast Inference via KonfAI Apps**
SlicerKonfAI runs KonfAI Apps directly inside Slicer, following the exact inference workflow defined in each App. This includes image preprocessing, model execution, and output reconstruction. The system supports GPU acceleration and optimized runtime performance to deliver fast, clinically usable inference.

### **• Quality Assurance (QA)**
SlicerKonfAI provides two complementary QA modules that assess prediction reliability by executing the App’s evaluation or uncertainty workflow, which can produce both quantitative metrics and qualitative output images:

- **Reference-based QA** : automatic comparison between predictions and available ground-truth annotations.
- **Reference-free QA**: uncertainty estimation when no reference is available, using:
  - TTA  
  - Stochastic dropout  
  - Multi-model ensembling

* **Tight integration with Slicer**
  Use volumes already loaded in Slicer (DICOM, NRRD, NIfTI, etc.) as inputs and write results back as:

  * Volume nodes (e.g. synthetic CT, logits, heatmaps)
  * Label maps / segmentation nodes (e.g. organ masks, tumor masks)

* **Configurable Apps root**  
  The workflows inside each App can be modified or extended (e.g., adding new evaluation metrics or custom processing steps), and SlicerKonfAI will automatically use the updated configuration.

* **Built-in State-of-the-Art Apps**
  By default, SlicerKonfAI includes several state-of-the-art KonfAI Apps for tasks such as anatomical segmentation and synthetic CT generation, providing ready-to-use baselines for experimentation and clinical evaluation.

---

SlicerKonfAI is built on top of the **KonfAI** , a fully configurable and modular deep learning framework that defines complete training, inference, and evaluation workflows through YAML files, enabling reproducible, transparent, and advanced medical imaging pipelines.

For more information about KonfAI, visit the project repository: https://github.com/vboussot/KonfAI

---

## 🚀 Installation

1. Install **3D Slicer ≥ 5.6**  
2. Clone this repository:
   ```bash
   git clone https://github.com/vboussot/SlicerKonfai.git
   ```
3. In Slicer, open:  
   **Edit → Application Settings → Modules → Additional Module Paths**  
   and add the folder `SlicerKonfai/KonfAI`
4. Restart Slicer and open the **KonfAI** module in the **Deep Learning** section.

## 🧩 What is a KonfAI App?

A **KonfAI App** is a self-contained workflow package built with KonfAI.  
It defines how a model is executed, how its outputs are generated, and how optional evaluation or uncertainty workflows are performed.  
Apps are portable, versioned, and can be executed identically from Python, the command line, or SlicerKonfAI.

A typical KonfAI App contains:

- **A trained model** (single checkpoint or ensemble)
- **Workflow configuration files** (`Prediction.yml`, `Evaluation.yml`, `Uncertainty.yml`) defining inference, evaluation, and uncertainty pipelines
- **A metadata file** (`app.json`) describing the App for SlicerKonfAI

A minimal App directory may look like:

```text
my_konfai_app/
├── app.json                # Metadata for SlicerKonfAI
├── Prediction.yml         # Inference config used by SlicerKonfAI
├── Evaluation.yml         # (Optional) evaluation workflow
├── Uncertainty.yml        # (Optional) uncertainty workflow
└── checkpoint.pt          # Checkpoint used by Prediction.yml
```

An example `app.json` could be:

```json
{
    "display_name": "Lung Lobe Segmentation",
    "short_description": "Deep learning model for segmenting lung lobes on CBCT scans.",
    "description": "This App performs domain adaptation by first synthesizing a CT-like volume from the input CBCT, followed by lung lobe segmentation using a 3D UNet-based model.",
    "tta": 4,
    "mc_dropout": 0
}
```

SlicerKonfAI uses this metadata to:

- Display the App name and description
- Enable App-specific options such as TTA or dropout

---

## ⚙️ How SlicerKonfAI runs an App (conceptual)

Internally, SlicerKonfAI typically:

1. Prepares a temporary working directory for the current case.
2. Exports the selected Slicer nodes to disk in MHA format, as expected by the App.
3. Executes the App by calling the KonfAI Apps CLI. For example, an inference call may look like:

   ```bash
   konfai-apps infer \
       -i Volume.mha \
       -o Output
       --ensemble 2
       --tta 2
       --mc 0
       --gpu 0
   ```
4. Monitors the running process (stdout/stderr) and streams logs to the Slicer interface.
5. Imports the generated outputs back into Slicer (volumes, segmentations, uncertainty maps, metrics).

--- 

> SlicerKonfAI = GUI + data exchange + process manager

> KonfAI = the engine that does all computations.

---
