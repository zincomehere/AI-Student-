import pandas as pd

df = pd.read_csv('StudentPerformanceFactors.csv')
print(f"😷 Dữ liệu đang bị nhiễm độc: {len(df)} dòng")

# Xóa sổ toàn bộ các dòng nhân bản vô tính
df_clean = df.drop_duplicates()

# Lưu lại trả về nguyên trạng
df_clean.to_csv('StudentPerformanceFactors.csv', index=False)
print(f"✨ Đã thanh tẩy thành công! Trả file gốc về lại: {len(df_clean)} dòng nguyên bản.")