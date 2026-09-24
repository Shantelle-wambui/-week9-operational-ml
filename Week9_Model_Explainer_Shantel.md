# Week 9 Model Explainer
## Equipment Failure Prediction — How the Model Works & When to Trust It

**Author:** Shantel Wambui  
**Course:** PLP Data Science & Machine Learning — Week 9  
**Date:** September 2026  

---

## The Analogy: How Does This Model Think?

Imagine you are an experienced senior mechanic who has serviced hundreds of machines over 20 years. When a machine comes in, you don't read one gauge and declare it healthy or failing. You check multiple things at once — the vibration feel, the heat coming off the motor, the oil pressure, how long since it was last serviced, and how old the machine is. After years of experience, you've built an internal pattern: *"When I see vibration this high AND the temperature is running hot AND it hasn't been serviced in 8 months — that machine is going to fail soon."*

**The XGBoost model is that experienced mechanic — but instead of intuition built over 20 years, it learned from 2,400 historical machine readings.** It checks all 9 sensor inputs simultaneously and assigns a failure probability score between 0% and 100%.

A score above **~50%** should be treated as a maintenance alert. A score above **80%** means the machine needs immediate attention.

---

## The Drivers: Top 3 Features Explained

The chart below (Feature Importance) shows which sensor readings the model relies on most when making its prediction.

*(See feature_importance.png — XGBoost panel)*

### 1. Vibration (mm/s) — Most Important Driver

Vibration is the single strongest predictor of imminent failure. When a machine's vibration reading exceeds approximately **3.5 mm/s**, the model sharply increases the failure probability. 

**Plain English:** A machine that's shaking more than usual is telling you something is wrong — a bearing wearing out, a shaft going out of balance, or a mounting loosening. The model has learned that high vibration readings precede ~35% of all failures in the training data.

### 2. Temperature (°C) — Second Driver

Operating temperature above **90°C** is the second strongest signal. Normal operating range is 65–85°C. Sustained high temperatures indicate inadequate cooling, friction buildup, or a blocked vent.

**Plain English:** A machine running hot is working too hard. Like a car engine overheating, sustained high temperature accelerates wear on seals, bearings, and lubricants. The model flags machines where temperature has crept above the safe operating band.

### 3. Maintenance Lag (days since last service) — Third Driver, Most Surprising

Machines that haven't been serviced for more than **300 days** show significantly elevated failure risk — even when their current sensor readings look acceptable.

**Plain English:** A machine that hasn't been looked at in almost a year may have slowly degrading components that sensors don't fully capture yet. Think of it like skipping your car's service: it may still drive fine, but the risk is accumulating quietly. The model learned that maintenance history is nearly as predictive as real-time sensor data.

---

## Trust & Limits: How the Operations Team Should Use This Tool

### What the model does well
- Catches machines with **multiple simultaneous warning signs** — it combines all 9 sensors in a way no single threshold alarm can
- Provides a **ranked risk list** so maintenance teams can prioritise which machines to check first when resources are limited
- Explains **why** a machine was flagged (via SHAP plots) — not just a black-box score

### Known Limitations

**False Positives (False Alarms):**  
The model will occasionally flag machines that turn out to be fine. A high-vibration reading could be from a temporary external source (nearby construction, a passing forklift). Before raising a work order, the technician should physically inspect the machine and cross-check with recent maintenance logs.

**False Negatives (Missed Failures):**  
Some failures happen suddenly without any gradual sensor drift — a sudden bearing crack, a power surge, a foreign object ingestion. The model cannot predict catastrophic instant failures. It is designed for **gradual degradation patterns** that develop over days to weeks.

**Sensor Data Quality:**  
The model is only as reliable as the sensor readings fed into it. A faulty vibration sensor reporting 0.0 mm/s will make a failing machine appear healthy. Regular sensor calibration is essential for the model to remain accurate.

### The Golden Rule
**Use this model as a guide, not a replacement for engineering judgment.**

The operations team should treat a high-risk score as a strong recommendation to prioritise a physical inspection — not as a guaranteed failure prediction. The goal is to reduce unplanned breakdowns by catching ~75% of failures 3–7 days early, while keeping unnecessary maintenance interventions to a manageable level.

A technician who disagrees with the model's assessment based on direct observation should always trust their hands-on judgment. The model augments human expertise — it does not replace it.

---

*Report generated for PLP Week 9 Assignment. Model: XGBoost with SMOTE. Dataset: 3,000 synthetic equipment sensor readings. ROC-AUC: see notebook evaluation section.*
