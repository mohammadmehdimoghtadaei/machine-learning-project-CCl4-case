import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from catboost import CatBoostRegressor
import seaborn as sns

# بارگذاری داده‌ها
data = pd.read_csv('filtered_data1.csv')

# انتخاب 50 ساختار برتر بر اساس Adsorption
top_50_data = data.nlargest(50, 'Adsorption')

# جدا کردن ویژگی‌ها و هدف (Adsorption)
X_top_50 = top_50_data.drop('Adsorption', axis=1)
y_top_50 = top_50_data['Adsorption']

# تقسیم به دو گروه: 25 ساختار بالایی و 25 ساختار پایینی از نظر Adsorption
group1 = X_top_50.iloc[:25]  # 25 ساختار با جذب بالاتر
group2 = X_top_50.iloc[25:]  # 25 ساختار با جذب پایین‌تر

# 1. محاسبه اهمیت ویژگی‌ها با مدل CatBoost
catboost_model = CatBoostRegressor(verbose=0, iterations=300,
        depth=8, learning_rate=0.05, colsample_bylevel=0.5,l2_leaf_reg=5, bagging_temperature=0.5,
        min_child_samples=10, subsample=0.7,
        random_state=42)  # تنظیم مدل CatBoost
catboost_model.fit(X_top_50, y_top_50)
feature_importances = catboost_model.get_feature_importance()

# ذخیره نتایج آزمون Mann-Whitney
p_values = []

# آستانه دلخواه برای p-value
threshold = 0.05  # اینجا مقدار دلخواه خود را وارد کنید

# اجرای آزمون Mann-Whitney برای هر ویژگی و ذخیره p-value
for feature in X_top_50.columns:
    stat, p_value = mannwhitneyu(group1[feature], group2[feature])
    p_values.append((feature, p_value))

# ایجاد DataFrame از نتایج p-value
p_value_df = pd.DataFrame(p_values, columns=['Feature', 'p-value']).sort_values(by='p-value')

# انتخاب ویژگی‌هایی که p-value آن‌ها کمتر از آستانه است
significant_features = p_value_df[p_value_df['p-value'] < threshold]

# نمایش ویژگی‌های معنی‌دار (p-value کمتر از آستانه)
print("Significant Features (p-value < threshold):\n", significant_features)

# ایجاد DataFrame برای اهمیت ویژگی‌ها
feature_importance_df = pd.DataFrame({
    'Feature': X_top_50.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# پرینت مقادیر اهمیت ویژگی‌ها
print("Feature Importances:\n", feature_importance_df)

# رسم نمودار ویژگی‌های با اهمیت (بدون گرید)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm')
plt.title('Feature Importance based on CatBoost Model', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(False)  # حذف گرید
plt.show()

# رسم نمودار میله‌ای برای نمایش p-value ها
plt.figure(figsize=(12, 8))
plt.barh(p_value_df['Feature'], p_value_df['p-value'], color='skyblue')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Significance Threshold ({threshold})')
plt.title('', fontsize=16)
plt.xlabel('p-value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()  # جلوگیری از هم‌پوشانی عناصر
plt.savefig('mann_analyze.svg', dpi=1200, bbox_inches='tight')
plt.grid(False)
plt.legend()
plt.show()
