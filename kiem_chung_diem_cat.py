import joblib

print("🔍 ĐANG KHỞI ĐỘNG MÁY X-QUANG GIẢI PHẪU CÂY QUYẾT ĐỊNH...\n")

# 1. Tải tệp mô hình đã được huấn luyện từ trước
try:
    surrogate_tree = joblib.load("surrogate_tree.pkl")
    print("✅ Đã load thành công tệp 'surrogate_tree.pkl' vào bộ nhớ!")
except Exception as e:
    print(f"❌ Không tìm thấy file: {e}")
    exit()

# Danh sách 11 thuộc tính y hệt lúc train
FEATURES = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
    'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality'
]

# 2. Truy cập vào cấu trúc lõi C++ của Scikit-learn
n_nodes = surrogate_tree.tree_.node_count
children_left = surrogate_tree.tree_.children_left
children_right = surrogate_tree.tree_.children_right
feature_indices = surrogate_tree.tree_.feature
thresholds = surrogate_tree.tree_.threshold

print("\n" + "="*50)
print("📊 BẢNG KÊ KHAI CÁC ĐIỂM CẮT TỬ THẦN DO AI TỰ HỌC ĐƯỢC")
print("="*50)

# 3. Quét qua toàn bộ ngã rẽ và trích xuất số liệu
diem_cat_theo_bien = {}

for i in range(n_nodes):
    # Nếu node này có chia nhánh (Không phải là lá)
    if children_left[i] != children_right[i]:
        ten_bien = FEATURES[feature_indices[i]]
        diem_cat = round(thresholds[i], 2)
        
        if ten_bien not in diem_cat_theo_bien:
            diem_cat_theo_bien[ten_bien] = set()
        diem_cat_theo_bien[ten_bien].add(diem_cat)

# 4. In bằng chứng thép ra màn hình
for bien, cac_diem_cat in diem_cat_theo_bien.items():
    danh_sach_so = ", ".join([str(so) for so in sorted(list(cac_diem_cat))])
    print(f"📍 Thuộc tính [{bien.ljust(20)}]: Cắt tại các mốc -> {danh_sach_so}")

print("\n💡 KẾT LUẬN: Mọi con số bạn thấy trên API đều được móc trực tiếp từ đây!")

