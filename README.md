# 📈 Natural Gas Price Estimation
> Time-series price forecasting for natural gas using **Polynomial Regression** with confidence intervals & a **FastAPI** estimator endpoint — inspired by the **JPMorgan Chase Quantitative Research** virtual experience.

---

## 📊 Dataset
| Field | Detail |
|---|---|
| Source | `Nat_Gas.csv` — monthly spot prices |
| Period | Oct 2020 → Aug 2024 (48 data points) |
| Price Range | **$9.84 – $12.80 / MMBtu** |
| Frequency | Month-end |

---

## ⚙️ Pipeline

**1. 📥 Data Ingestion** — parse dates, set datetime index, sort, assert zero nulls

**2. 🔧 Feature Engineering** — encode time as integer `t` (months since origin) for regression

**3. 🔍 Outlier Detection** — Z-score threshold (`|z| > 2.5`) to flag anomalous prices

**4. 📐 Polynomial Regression (Degree 2)**
- Train/test split at `t = 35` (Apr 2024)
- `sklearn` Pipeline: `PolynomialFeatures → LinearRegression`
- Refit on full dataset for forecasting

**5. 🔮 12-Month Forecast** — Oct 2024 → Sep 2025 with **expanding 95% confidence intervals**
```
CI = 1.96 × σ × (1 + 0.05 × max(0, t − 47))
```

**6. ✅ Cross-Validation** — `TimeSeriesSplit` (5 folds) on degree-2 polynomial

**7. 🌐 FastAPI Endpoint** — `/estimate?date=YYYY-MM-DD` returns price + CI bounds

**8. 📊 Visualisation** — actual scatter · fitted curve · forecast + shaded CI band

---

## 🔮 Sample Forecast Output
```json
{
  "date": "2025-03-31",
  "estimate": 12.XXXX,
  "ci_lower": XX.XXXX,
  "ci_upper": XX.XXXX
}
```

---

## 🛠️ Tech Stack
`Python 3.13` · `pandas` · `numpy` · `scikit-learn` · `scipy` · `matplotlib` · `FastAPI`

---

## 🚀 Quickstart
```bash
pip install pandas numpy scikit-learn scipy matplotlib fastapi uvicorn
jupyter notebook Nat_gas_estimation.ipynb
```

> 📁 Update the CSV path in cell 2 to point to your local `Nat_Gas.csv`

> 🌐 To run the API: `uvicorn app:app --reload`, then hit `/estimate?date=2025-06-30`
