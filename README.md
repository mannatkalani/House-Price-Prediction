# 🏠 House Price Prediction

A machine learning web application that predicts house prices using a **Linear Regression** model trained on King County, WA real estate data. Built with **Streamlit** and **Plotly** for an interactive, professional dashboard experience.

---

## 📸 Features

- 🔮 **Live Price Prediction** — Adjust property inputs and get an instant predicted price
- 📊 **Interactive Charts** — Price distribution, scatter plots, bar charts, and feature coefficient analysis
- 🏙️ **Market Insights** — Top cities by average price, condition impact, bedroom breakdowns
- 📐 **Model Transparency** — Visualize how each feature influences the predicted price
- ⚡ **Cached Performance** — Data and model are cached for fast, responsive interactions

---

## 🗂️ Project Structure

```
House-Price-Prediction/
├── app.py          # Main Streamlit application
├── house.pkl       # Trained LinearRegression model (joblib format)
├── data.csv        # King County housing dataset (4,600 records)
└── README.md
```

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone https://github.com/mannatkalani/House-Price-Prediction.git
cd House-Price-Prediction
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install streamlit pandas numpy plotly joblib scikit-learn
```

---

## 🚀 Usage

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Library | scikit-learn 1.6.1 |
| Serialization | joblib |
| Training Data | King County, WA |
| Records | 4,600 |
| Target Variable | `price` (USD) |

**Input Features:**

| Feature | Description |
|---|---|
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Interior living area (sqft) |
| `sqft_lot` | Lot size (sqft) |
| `floors` | Number of floors |
| `waterfront` | Waterfront property (0/1) |
| `view` | View quality score (0–4) |
| `condition` | Property condition (1–5) |
| `sqft_above` | Above-ground area (sqft) |
| `sqft_basement` | Basement area (sqft) |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year last renovated (0 = never) |

---

## 📊 Dataset

The dataset contains **4,600 house sales** from King County, Washington (Seattle area), with prices ranging from under \$100K to over \$2.6M.

Key columns: `date`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `street`, `city`, `statezip`, `country`

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** — Web app framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **scikit-learn** — ML model
- **joblib** — Model serialization

---

## 👤 Author

**Mannat Kalani**  
GitHub: [@mannatkalani](https://github.com/mannatkalani)
