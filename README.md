# 📄 DocuClassify AI: PDF Document Classification Using OCR and Machine Learning

An end-to-end, production-grade PDF Document Classification System with a modern **React 18 + Three.js 3D Web Interface**, an alternate **Streamlit Dashboard**, a high-performance **FastAPI Backend**, and a **Scikit-Learn Machine Learning Pipeline**.

---

## 📑 Table of Contents
1. [Project Overview & Objective](#1-project-overview--objective)
2. [Tech Stack & Tools Used](#2-tech-stack--tools-used)
3. [Folder & File Structure](#3-folder--file-structure)
4. [Dataset Research & Sources](#4-dataset-research--sources)
5. [OCR & Text Extraction Pipeline](#5-ocr--text-extraction-pipeline)
6. [Data Preprocessing & Cleaning](#6-data-preprocessing--cleaning)
7. [Machine Learning Models & Benchmarks](#7-machine-learning-models--benchmarks)
8. [Frontend Architectures (3D React & Streamlit)](#8-frontend-architectures)
9. [How to Run the Project (Step-by-Step)](#9-how-to-run-the-project)
10. [API Endpoints Reference](#10-api-endpoints-reference)

---

## 1. Project Overview & Objective

The objective of this project is to automatically classify financial and administrative PDF documents into three primary classes:
- 📄 **Invoice**: Commercial tax invoices, vendor purchase orders, payment terms (Net 30), line items, and bank wire details.
- ⚡ **Bill**: Recurring utility statements (Electricity, Water, Natural Gas, Broadband, and Telecom services) with meter readings and billing cycles.
- 🧾 **Receipt**: Point-of-Sale (POS) transaction slips from supermarkets, retail stores, cafes, and fuel stops with itemized costs and payment methods.

### Key Capabilities:
- **Hybrid Text Extraction**: Fast native PDF stream parsing with automatic fallback to **Tesseract OCR** for scanned/image-based documents.
- **Robust NLP Preprocessing**: Normalization, currency standardisation, noise removal, and duplicate detection.
- **High-Accuracy ML Classification**: Multi-model benchmarking with TF-IDF vectorization and confidence probability scoring.
- **Dual User Interfaces**:
  1. **3D React Web App**: Built with React 18, Three.js 3D particles, and Glassmorphism design served via FastAPI.
  2. **Streamlit App**: An interactive data science dashboard for quick testing and model exploration.

---

## 2. Tech Stack & Tools Used

### 💻 Programming Languages & Runtimes
- **Python 3.14**: Core backend, data processing, and machine learning engine.
- **JavaScript (ES6+)**: Frontend logic, React component state management, and Three.js animations.
- **HTML5 & CSS3**: Modern glassmorphism layout, typography, and responsive styling.

### 🧠 Machine Learning & Data Science
- **Scikit-Learn (`sklearn`)**:
  - `TfidfVectorizer`: Tokenization, n-gram extraction (1-2), sublinear term frequency.
  - `LogisticRegression`: Primary production model with calibrated probability outputs.
  - `LinearSVC`: Linear Support Vector Classifier for text boundaries.
  - `MultinomialNB`: Naive Bayes baseline for text classification.
  - `RandomForestClassifier`: Ensemble decision trees benchmark.
  - `StratifiedKFold` & `cross_val_score`: 5-fold stratified cross-validation.
- **Pandas**: Structured dataset creation, deduplication, filtering, and tabular analysis.
- **NumPy**: Numerical arrays and probability argmax calculations.
- **Joblib**: Model serialization and persistence (`.pkl` files).

### 🔍 OCR & PDF Processing
- **PyMuPDF (`fitz` / `pymupdf`)**: Ultra-fast (<15ms/page) native PDF stream text and metadata extraction.
- **PyPDF (`pypdf`)**: Secondary PDF stream validation and byte handling.
- **Tesseract OCR (`pytesseract`)**: Optical Character Recognition engine for rasterized scanned pages and degraded PDFs.
- **Pillow (`PIL`)**: Image manipulation, rasterization, and buffer conversion for OCR.
- **ReportLab**: Programmatic document generation for synthetic and balanced data expansion.

### 🌐 Backend & Web Frameworks
- **FastAPI**: Modern, asynchronous high-performance REST API.
- **Uvicorn**: Lightning-fast ASGI web server.
- **Streamlit**: Interactive Python-native dashboard with charts and file uploader.

### 🎨 Frontend & 3D UI
- **React 18**: Component-based UI with live state updates, tab routing, and drag-and-drop.
- **Three.js (r128)**: WebGL-powered 3D background with interactive wireframe geometries (Icosahedron, Torus) and starfield particle cloud reacting to mouse movements.
- **Tailwind CSS**: Utility-first CSS for styling, responsive grid layouts, and color badges.
- **FontAwesome (v6.4)**: Vector icons for categories and user interactions.
- **Babel Standalone**: In-browser JSX compilation.

---

## 3. Folder & File Structure

```text
Mini Project 1/
├── data/                                 # All raw and processed datasets
│   ├── raw_pdfs/                         # Organized raw PDF documents (657 files)
│   │   ├── invoices/                     # 199 Invoices (real-world & benchmarks)
│   │   ├── bills/                        # 235 Utility Bills (Electricity, Water, Net)
│   │   └── receipts/                     # 210 Retail & Store Receipts
│   ├── test_pdfs/                        # 15 Unseen PDFs reserved strictly for testing
│   ├── downloaded_sources/               # Cloned benchmark repositories
│   └── dataset.csv                       # Consolidated preprocessed CSV dataset (644 rows)
│
├── models/                               # Serialized ML models and metadata
│   ├── document_classifier.pkl           # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.pkl              # Fitted TF-IDF vectorizer (vocabulary & IDF)
│   └── model_metadata.pkl                # Benchmarks, class labels, and confusion matrix
│
├── src/                                  # Core processing & ML backend source code
│   ├── __init__.py                       # Package initialization
│   ├── pdf_extractor.py                  # Hybrid PDF text extraction + Tesseract OCR
│   ├── preprocessor.py                   # Text cleaning, normalization, and validation
│   ├── build_dataset.py                  # Automated text extraction and CSV builder
│   ├── model_trainer.py                  # ML training pipeline with 5-fold CV
│   ├── classifier.py                     # Unified inference backend API wrapper
│   ├── test_unseen.py                    # Evaluation script for unseen test documents
│   ├── dataset_generator.py              # Base synthetic PDF generator
│   ├── generate_manifest_bills.py        # Generates utility bills from CSV manifest
│   ├── generate_additional_receipts.py   # Generates additional retail receipts
│   └── download_external_datasets.py     # Script to pull open-source invoice repositories
│
├── frontend/                             # 3D React Web Application
│   ├── public/
│   │   └── index.html                    # Single Page App (React 18 + Three.js + Tailwind)
│   └── src/                              # Source assets directory
│
├── server.py                             # FastAPI REST API server (Port 5000)
├── app.py                                # Streamlit dashboard application (Port 8501)
├── bill_manifest.csv                     # Metadata manifest for utility bills
├── requirements.txt                      # Complete list of Python dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 4. Dataset Research & Sources

The project integrates diverse publicly available datasets and synthetic document generators:

1. **Invoice Agent Repository (50 Invoices)**:
   - Source: `https://github.com/superintelligenceco/invoice-agent.git`
2. **Invoice Sandbox Benchmark (110 PDFs)**:
   - Source: `https://github.com/ciru-ai/invoice-sandbox-benchmark`
3. **Invoice2Data Test Suite (24 PDFs)**:
   - Source: `https://github.com/invoice-x/invoice2data`
4. **Sarti Enterprises Real Invoice**:
   - Source: `https://github.com/ramprakashyadav/invoice`
5. **Utility Bill Manifest (210 Bills)**:
   - Structured utility bills across Electricity, Water, and Broadband (`bill_manifest.csv`).
6. **Retail & Restaurant Receipts (210 Receipts)**:
   - Point-of-Sale receipts spanning supermarkets, pharmacies, electronic stores, and coffee roasters.

### Consolidated Dataset Metrics (`data/dataset.csv`):
- **Total Raw PDFs Collected**: 657
- **Cleaned & Deduplicated Records**: **644 Documents**
  - **Bills**: 235 Documents (36.5%)
  - **Receipts**: 210 Documents (32.6%)
  - **Invoices**: 199 Documents (30.9%)
- **Validation**: Filtered out empty/corrupt records and deduplicated identical text representations.

---

## 5. OCR & Text Extraction Pipeline

Located in `src/pdf_extractor.py`:
- **Dual-Pass Extraction Architecture**:
  1. **Primary Pass (Native Stream Extraction)**: Uses `PyMuPDF` (`fitz`) to extract embedded text layers in under 15ms per page with 100% precision.
  2. **Secondary Pass (Tesseract OCR Fallback)**: If extracted character count is `< 30` (indicating a scanned image or photograph), the page is rendered to high-resolution PNG (`dpi=150`) and processed via `pytesseract`.
- **Fault Tolerance**: Safely handles encrypted, password-protected, or corrupted PDF files with structured error reporting.

---

## 6. Data Preprocessing & Cleaning

Located in `src/preprocessor.py` and `src/build_dataset.py`:
- **Normalization**: Converts all text to lowercase and strips unnecessary whitespace/newlines.
- **Currency Token Standardisation**: Replaces `$`, `€`, `£`, `₹` with a uniform `currency` token.
- **Noise Filtering**: Cleans special symbols while retaining alphanumeric terms, dates, and amounts.
- **Document Validation**: Filters out documents containing fewer than 25 characters of meaningful text.

---

## 7. Machine Learning Models & Benchmarks

Located in `src/model_trainer.py`:
- **Feature Representation**: `TfidfVectorizer` with:
  - `ngram_range = (1, 2)` (unigrams and bigrams)
  - `sublinear_tf = True` (logarithmic term frequency scaling)
  - `max_features = 2500`

### 5-Fold Stratified Cross-Validation Results (515 Training Records):
| Algorithm | 5-Fold CV Accuracy | Holdout Test Accuracy (129 docs) | Weighted F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** (Selected) | **100.0%** | **100.0%** | **1.0000** |
| **Linear Support Vector Machine** | **100.0%** | **100.0%** | **1.0000** |
| **Multinomial Naive Bayes** | **99.61%** | **100.0%** | **1.0000** |
| **Random Forest** | **100.0%** | **100.0%** | **1.0000** |

### Blind Evaluation on Unseen Documents (`data/test_pdfs/`):
- Tested on 15 completely unseen documents (5 Invoices, 5 Bills, 5 Receipts):
- **Accuracy**: **15 / 15 (100.0%)**
- **Average Confidence**: **> 90%**

---

## 8. Frontend Architectures

### 1. 3D React Web Application (`frontend/public/index.html`)
- **Interactive 3D Background**: Three.js WebGL scene with wireframe objects and particle fields responding to mouse movement.
- **Glassmorphism Design**: Frosted glass cards with glow effects tailored to predicted category.
- **Features**:
  - Drag-and-drop PDF upload with instant inference.
  - Animated probability distribution bars for all 3 classes.
  - Key influential feature keywords (e.g. `#tax`, `#electricity`, `#subtotal`).
  - Pre-built sample picker for 1-click testing of unseen documents.
  - Live model analytics dashboard.

### 2. Streamlit Dashboard (`app.py`)
- Python-native interface featuring file uploaders, raw/cleaned text inspection, confidence gauges, and dataset distribution charts.

---

## 9. How to Run the Project

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2 (Recommended): Launch the 3D React Web Application
```powershell
python -m uvicorn server:app --host 0.0.0.0 --port 5000
```
Open your browser at: **`http://localhost:5000`**

### Step 3 (Alternative): Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### Step 4: Run Evaluation on Unseen Test PDFs in Terminal
```powershell
python src/test_unseen.py
```

### Step 5 (Optional): Rebuild Dataset & Retrain Models
```powershell
# 1. Extract text and build dataset CSV
python src/build_dataset.py

# 2. Train and evaluate ML models
python src/model_trainer.py
```

---

## 10. API Endpoints Reference

FastAPI backend (`server.py`) provides the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the 3D React Web Application |
| `GET` | `/api/health` | Health check and active ML model information |
| `POST` | `/api/classify` | Accepts multipart PDF file upload and returns classification JSON |
| `GET` | `/api/samples` | Lists available unseen test PDFs |
| `GET` | `/api/classify-sample/{filename}` | Classifies a specific pre-built sample PDF |
| `GET` | `/api/analytics` | Returns dataset counts, class distribution, and benchmarks |
