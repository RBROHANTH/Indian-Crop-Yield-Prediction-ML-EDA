# ============================================================
#   CROP YIELD PREDICTION - INDIA (1990-2013)
#   Author: Rohanth R B
#   Dataset: India crop yield data (4048 records, 8 crops)
#   Goal: Predict crop yield (hg/ha) using rainfall, temp,
#         pesticides, year, and crop type
# ============================================================

# ── STEP 0: INSTALL (run once if needed) ────────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost


# ── STEP 1: IMPORTS ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: XGBoost (better performance, worth installing)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120


# ────────────────────────────────────────────────────────────
# PHASE 1: DATA LOADING & EXPLORATION
# ────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("=" * 60)

# Load
df = pd.read_csv('india_yield_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)  # drop index column

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nCrops in dataset: {df['Item'].unique()}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")


# ────────────────────────────────────────────────────────────
# PHASE 2: DATA VISUALIZATION
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2: DATA VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Crop Yield Analysis - India (1990–2013)", fontsize=16, fontweight='bold', y=1.01)

# ── Plot 1: Average yield per crop (bar chart) ───────────────
avg_yield = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False)
axes[0, 0].bar(avg_yield.index, avg_yield.values, color=sns.color_palette("muted", len(avg_yield)))
axes[0, 0].set_title("Average Yield by Crop (hg/ha)", fontweight='bold')
axes[0, 0].set_xlabel("Crop")
axes[0, 0].set_ylabel("Yield (hg/ha)")
axes[0, 0].tick_params(axis='x', rotation=30)

# ── Plot 2: Yield trend over years (line chart per crop) ─────
for crop in df['Item'].unique():
    subset = df[df['Item'] == crop].groupby('Year')['hg/ha_yield'].mean()
    axes[0, 1].plot(subset.index, subset.values, marker='o', markersize=3, label=crop)
axes[0, 1].set_title("Yield Trend Over Years (per Crop)", fontweight='bold')
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Avg Yield (hg/ha)")
axes[0, 1].legend(fontsize=6, loc='upper left')

# ── Plot 3: Rainfall vs Yield (scatter) ──────────────────────
axes[0, 2].scatter(df['average_rain_fall_mm_per_year'], df['hg/ha_yield'],
                   alpha=0.3, color='steelblue', edgecolors='none', s=15)
axes[0, 2].set_title("Rainfall vs Yield", fontweight='bold')
axes[0, 2].set_xlabel("Avg Rainfall (mm/year)")
axes[0, 2].set_ylabel("Yield (hg/ha)")

# ── Plot 4: Temperature vs Yield (scatter) ───────────────────
axes[1, 0].scatter(df['avg_temp'], df['hg/ha_yield'],
                   alpha=0.3, color='tomato', edgecolors='none', s=15)
axes[1, 0].set_title("Temperature vs Yield", fontweight='bold')
axes[1, 0].set_xlabel("Avg Temperature (°C)")
axes[1, 0].set_ylabel("Yield (hg/ha)")

# ── Plot 5: Pesticides vs Yield (scatter) ────────────────────
axes[1, 1].scatter(df['pesticides_tonnes'], df['hg/ha_yield'],
                   alpha=0.3, color='mediumseagreen', edgecolors='none', s=15)
axes[1, 1].set_title("Pesticides vs Yield", fontweight='bold')
axes[1, 1].set_xlabel("Pesticides Used (tonnes)")
axes[1, 1].set_ylabel("Yield (hg/ha)")

# ── Plot 6: Correlation Heatmap ───────────────────────────────
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2],
            linewidths=0.5, square=True)
axes[1, 2].set_title("Feature Correlation Heatmap", fontweight='bold')

plt.tight_layout()
plt.savefig("viz_1_eda.png", bbox_inches='tight')
plt.show()
print("Saved: viz_1_eda.png")


# ── Yield distribution by crop (boxplot) ─────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
crop_order = df.groupby('Item')['hg/ha_yield'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Item', y='hg/ha_yield', order=crop_order,
            palette="Set2", ax=ax)
ax.set_title("Yield Distribution by Crop Type", fontsize=14, fontweight='bold')
ax.set_xlabel("Crop")
ax.set_ylabel("Yield (hg/ha)")
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig("viz_2_yield_distribution.png", bbox_inches='tight')
plt.show()
print("Saved: viz_2_yield_distribution.png")


# ────────────────────────────────────────────────────────────
# PHASE 3: DATA PREPROCESSING
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3: DATA PREPROCESSING")
print("=" * 60)

# Encode crop names to numbers (Label Encoding)
le = LabelEncoder()
df['crop_encoded'] = le.fit_transform(df['Item'])

print(f"\nCrop encoding mapping:")
for crop, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {crop:20s} → {code}")

# Define features and target
features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'crop_encoded']
target   = 'hg/ha_yield'

X = df[features]
y = df[target]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ────────────────────────────────────────────────────────────
# PHASE 4: MODEL TRAINING
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 4: MODEL TRAINING")
print("=" * 60)

results = {}

# ── Model 1: Linear Regression (baseline) ───────────────────
print("\n[1/3] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = {
    'R2' : r2_score(y_test, y_pred_lr),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'y_pred': y_pred_lr
}

# ── Model 2: Random Forest ───────────────────────────────────
print("[2/3] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)   # tree models don't need scaling
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'R2' : r2_score(y_test, y_pred_rf),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'y_pred': y_pred_rf
}

# ── Model 3: XGBoost (if available) ─────────────────────────
if XGBOOST_AVAILABLE:
    print("[3/3] Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42,
                        verbosity=0, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = {
        'R2' : r2_score(y_test, y_pred_xgb),
        'MAE': mean_absolute_error(y_test, y_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'y_pred': y_pred_xgb
    }
else:
    print("[3/3] Skipped XGBoost (not installed)")

# ── Print comparison table ───────────────────────────────────
print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(f"{'Model':<22} {'R² Score':>10} {'MAE':>12} {'RMSE':>14}")
print("-" * 60)
for model_name, metrics in results.items():
    print(f"{model_name:<22} {metrics['R2']:>10.4f} {metrics['MAE']:>12.2f} {metrics['RMSE']:>14.2f}")


# ────────────────────────────────────────────────────────────
# PHASE 5: RESULTS VISUALIZATION
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 5: RESULTS VISUALIZATION")
print("=" * 60)

# ── Best model = Random Forest (pick highest R2) ─────────────
best_model_name = max(results, key=lambda m: results[m]['R2'])
best_pred = results[best_model_name]['y_pred']
print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Model Results — {best_model_name}", fontsize=14, fontweight='bold')

# ── Plot A: Actual vs Predicted ──────────────────────────────
axes[0].scatter(y_test, best_pred, alpha=0.4, color='royalblue', s=15)
min_v, max_v = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Perfect fit')
axes[0].set_title("Actual vs Predicted Yield")
axes[0].set_xlabel("Actual Yield (hg/ha)")
axes[0].set_ylabel("Predicted Yield (hg/ha)")
axes[0].legend()

# ── Plot B: Residuals ────────────────────────────────────────
residuals = y_test.values - best_pred
axes[1].scatter(best_pred, residuals, alpha=0.4, color='coral', s=15)
axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1].set_title("Residual Plot")
axes[1].set_xlabel("Predicted Yield (hg/ha)")
axes[1].set_ylabel("Residuals")

# ── Plot C: Feature Importance (Random Forest) ───────────────
if 'Random Forest' in results:
    importance = rf.feature_importances_
    feat_names  = features
    sorted_idx  = np.argsort(importance)[::-1]
    axes[2].bar([feat_names[i] for i in sorted_idx],
                [importance[i] for i in sorted_idx],
                color=sns.color_palette("muted", len(features)))
    axes[2].set_title("Feature Importance (Random Forest)")
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("Importance Score")
    axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("viz_3_model_results.png", bbox_inches='tight')
plt.show()
print("Saved: viz_3_model_results.png")


# ── Model comparison bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
model_names = list(results.keys())
r2_scores   = [results[m]['R2'] for m in model_names]
colors = ['#4C72B0', '#55A868', '#C44E52'][:len(model_names)]
bars = ax.bar(model_names, r2_scores, color=colors, width=0.5)
for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{score:.4f}", ha='center', fontsize=11, fontweight='bold')
ax.set_title("R² Score Comparison Across Models", fontsize=13, fontweight='bold')
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("viz_4_model_comparison.png", bbox_inches='tight')
plt.show()
print("Saved: viz_4_model_comparison.png")


# ────────────────────────────────────────────────────────────
# PHASE 6: CROSS-VALIDATION (Robustness Check)
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 6: CROSS-VALIDATION (5-Fold)")
print("=" * 60)

cv_rf = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        X, y, cv=5, scoring='r2')
print(f"\nRandom Forest 5-Fold CV R² scores: {cv_rf.round(4)}")
print(f"Mean R²: {cv_rf.mean():.4f}  |  Std: {cv_rf.std():.4f}")


# ────────────────────────────────────────────────────────────
# PHASE 7: PREDICTION ON NEW DATA (Demo)
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 7: PREDICTION ON NEW SAMPLE")
print("=" * 60)

# Predict yield for Rice in 2010 with typical Indian conditions
sample = pd.DataFrame({
    'Year'                          : [2010],
    'average_rain_fall_mm_per_year' : [1200],
    'pesticides_tonnes'             : [50000],
    'avg_temp'                      : [26.0],
    'crop_encoded'                  : [le.transform(['Rice, paddy'])[0]]
})

pred_yield = rf.predict(sample)[0]
print(f"\nSample Input:")
print(f"  Crop       : Rice, paddy")
print(f"  Year       : 2010")
print(f"  Rainfall   : 1200 mm/year")
print(f"  Pesticides : 50,000 tonnes")
print(f"  Avg Temp   : 26.0 °C")
print(f"\nPredicted Yield: {pred_yield:,.0f} hg/ha  ({pred_yield/100:,.0f} kg/ha)")


# ────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"""
Dataset      : India Crop Yield (1990-2013)
Records      : 4,048 | Crops: 8 | Features: 5
Target       : Yield in hg/ha

Best Model   : {best_model_name}
R² Score     : {results[best_model_name]['R2']:.4f}
MAE          : {results[best_model_name]['MAE']:,.2f} hg/ha
RMSE         : {results[best_model_name]['RMSE']:,.2f} hg/ha

Key Insight  : Crop type is the dominant predictor of yield.
               Pesticide use and temperature show moderate correlation.
               Rainfall alone is a weak predictor across all crops.

Output files : viz_1_eda.png
               viz_2_yield_distribution.png
               viz_3_model_results.png
               viz_4_model_comparison.png
""")
