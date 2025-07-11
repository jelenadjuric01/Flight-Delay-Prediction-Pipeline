# ✈️ Flight-Delay Prediction with Apache Spark

Predict the **arrival delay** of U.S. domestic flights using a machine-learning model built and served with Apache Spark.  
The project was developed for the *Big Data – Spark Practical Work* assignment (First semester 2024/25).

---

## 📁 Repository layout

| Path / file            | Purpose                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `notebook.ipynb`       | Exploration, cleaning, feature engineering, model selection & training |
| `app.py`               | Stand-alone Spark app that loads **any** test CSVs, applies the saved model and prints metrics |
| `best_model/`          | Serialized `CrossValidatorModel` produced in the notebook     |
| `report.pdf`           | Technical report summarising methodology & results |
| `requirements.txt`     | (Optional) extra Python libraries not bundled with Spark      |
| `training_data/`       | **Place your own** CSVs for model development here            |
| `results/`             | Auto-generated folder containing `results.txt` after running `app.py` |

---

## ⚙️ Prerequisites

| Tool | Tested version |
| ---- | -------------- |
| Java | 8 or 11        |
| Apache Spark | 3.5.x (Hadoop 3) |
| Python | 3.9 – 3.11   |
| pip packages | See `requirements.txt` (mainly `pyspark`) |

> **Tip:** On a fresh machine run  
> `pip install -r requirements.txt`

---

## 🚀 Quick start

### 1 · Clone and install

```bash
git clone <https://github.com/jelenadjuric01/Flight-Delay-Prediction-Pipeline>
cd flight-delay-spark
pip install -r requirements.txt   # if provided
```

### 2 · Prepare data

Download one or more monthly CSV files from the U.S. DOT dataset  
and drop them into `training_data/`. Only the columns allowed by the assignment are kept automatically.

### 3 · Train & validate in the Notebook

```bash
jupyter lab  # or jupyter notebook
# open notebook.ipynb and run all cells
```

* The notebook cleans the data, engineers carrier/airport average-delay features, normalises all inputs,  
  performs **5-fold cross-validation** over four algorithms, and saves the best model to `best_model/`.  
* Best CV score (validation RMSE): **16.13 minutes** using a **Linear Regression** with elastic-net = 0 and five selected features.

### 4 · Test on unseen data

```bash
spark-submit --master local[*] app.py /path/to/test_csv_folder /path/to/best_model
```

* Prints the first 20 predictions and three metrics: RMSE, MAE, R².  
* Writes the same metrics plus the first 100 predictions to `results/results.txt`.  
* Example result on `2008.csv` + `1989.csv`: **RMSE ≈ 31.3 min, MAE ≈ 15.7 min, R² ≈ 0.27**.

---

## 🧠 Model details

* **Input features (after selection)**  
  `Month`, `DepDelay`, `Dest_Avrg_Delay`, `Carrier_Avrg_Delay`, `Origin_Avrg_Delay`,  
  `DayOfWeek`, `DayofMonth`
* **Algorithm** Elastic-Net Linear Regression (`elasticNetParam=0`, `regParam=0.05`, `maxIter=10`)
* **Pre-processing** 
  * Forbidden columns dropped  
  * Missing / “NA” values removed  
  * Negative delays filtered out  
  * Min-max scaling to `[0,1]`

For full rationale see `report.pdf`.

---

## 🔄 Re-training with different data

1. Replace or append new CSV files in `training_data/`.  
2. Re-run **only the “Data Loading ➜ Model Training”** cells in the notebook.  
3. Commit the new `best_model/` folder.

The downstream `app.py` script needs **no changes**.

---

## 📝 How to cite data

Flight data © U.S. Department of Transportation, accessed via Harvard Dataverse  
(doi: 10.7910/DVN/HG7NV7).

---

## 👥 Authors & license

*Group 4 – Big Data 2024/25*  
See `report.pdf` for individual contributions.

Project code is released under the MIT License (see `LICENSE`).
