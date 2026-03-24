import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
from catboost import CatBoostClassifier
import os

print("🚀 ĐANG TẠO CÂY QUYẾT ĐỊNH TỔNG QUÁT VÀ LƯU THÀNH FILE...")

if not os.path.exists('StudentPerformanceFactors.csv'):
    print("❌ LỖI: Không tìm thấy file StudentPerformanceFactors.csv trong thư mục này!")
    print("Hãy copy file CSV vào đây rồi chạy lại nhé.")
    exit()

df = pd.read_csv('StudentPerformanceFactors.csv')
features = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
    'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality'
]

# Xử lý khuyết thiếu nhanh
for col in df.select_dtypes(include=['number']).columns: df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude=['number']).columns: df[col] = df[col].fillna(df[col].mode()[0])

encoders = joblib.load("label_encoders.pkl")
for col in encoders:
    if col in df.columns:
        df[col] = df[col].astype(str)
        mapping_dict = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
        df[col] = df[col].map(mapping_dict).fillna(0).astype(int)
        
X = df[features]

# Lấy Não CatBoost ra dự đoán
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")
catboost_predictions = model.predict(X)

# Tạo cây độ sâu 3
surrogate_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
surrogate_tree.fit(X, catboost_predictions)

# LƯU CÂY VÀO FILE .PKL 
joblib.dump(surrogate_tree, "surrogate_tree.pkl")

# LƯU FILE TEXT 
tree_rules = export_text(surrogate_tree, feature_names=features)
tree_rules = tree_rules.replace("class: 0", "Dự báo: AN TOÀN").replace("class: 1", "Dự báo: RỦI RO CAO")
with open("cay_tong_quat.txt", "w", encoding="utf-8") as f:
    f.write(tree_rules)

print("✅ NGON LÀNH! Đã đẻ ra 2 file: surrogate_tree.pkl và cay_tong_quat.txt")