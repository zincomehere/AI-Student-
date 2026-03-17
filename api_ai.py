# ==============================================================================
# YÊU CẦU CÀI ĐẶT THƯ VIỆN BẰNG TERMINAL (VS Code):
# pip install fastapi uvicorn pandas catboost scikit-learn joblib pydantic
#
# LỆNH KHỞI ĐỘNG SERVER (Mở Terminal tại thư mục chứa file này):
# uvicorn api_ai:app --reload
# ==============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# 1. KHỞI TẠO APP VÀ BẢO MẬT CORS (Cho phép Frontend Web gọi API thoải mái)
app = FastAPI(title="Hệ thống Cảnh báo Rủi ro Sinh viên - AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các tên miền gọi API (Dễ dàng khi test)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. KHAI BÁO BIẾN TOÀN CỤC (Lưu Model và Encoder)
model = None
encoders = None

# Danh sách 11 đặc trưng (Phải khớp 100% thứ tự các cột lúc Train AI)
FEATURES = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
    'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality'
]

# Khuôn mẫu dữ liệu BE phải gửi qua (Pydantic sẽ tự động bắt lỗi nếu BE gửi thiếu hoặc sai)
class StudentData(BaseModel):
    Attendance: float
    Hours_Studied: float
    Previous_Scores: float
    Access_to_Resources: str
    Motivation_Level: str
    Family_Income: str
    Peer_Influence: str
    Distance_from_Home: str
    Extracurricular_Activities: str
    Sleep_Hours: float
    Teacher_Quality: str

# 3. NẠP MÔ HÌNH VÀO RAM KHI BẬT SERVER (CHỈ CHẠY 1 LẦN DUY NHẤT)
@app.on_event("startup")
def load_ai_assets():
    global model, encoders
    print("🚀 Đang khởi động Não AI (CatBoost) và tải Từ điển...")
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        encoders = joblib.load("label_encoders.pkl")
        print("✅ Nạp thành công! API đã sẵn sàng nhận Request ở cổng 8000.")
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file .cbm hoặc .pkl. Hãy kiểm tra lại thư mục! Chi tiết: {e}")

# 4. API TEST SỨC KHỎE (Để kiểm tra xem AI có đang chạy không)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server AI đang hoạt động rất khỏe!"}

# 5. LÕI API: DỰ BÁO VÀ PHÂN TÍCH LÝ DO CẢNH BÁO
@app.post("/api/predict")
def predict_risk(student: StudentData):
    try:
        # 5.1. Đóng gói dữ liệu đầu vào thành DataFrame
        data_dict = student.dict()
        df_input = pd.DataFrame([data_dict])
        
        # 5.2. Tiền xử lý: Dịch Label Encoding an toàn
        for col in encoders:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
                try:
                    df_input[col] = encoders[col].transform(df_input[col])
                except ValueError:
                    # Rủi ro: Có nhãn từ vựng lạ chưa từng học -> Mặc định gán về 0 (nhãn phổ biến nhất) để không sập AI
                    df_input[col] = 0 
                
        # Ép chuẩn thứ tự cột trước khi ném vào CatBoost
        X_input = df_input[FEATURES]
        
        # 5.3. Gọi AI dự báo xác suất (Lấy xác suất rớt môn - Index 1)
        risk_probability = model.predict_proba(X_input)[:, 1][0] * 100
        risk_probability = round(risk_probability, 2)
        
        # 5.4. Phân loại theo Ngưỡng rủi ro
        if risk_probability >= 65:
            risk_level = "CAO (Nguy hiểm)"
        elif risk_probability >= 40:
            risk_level = "TRUNG BÌNH (Cần theo dõi)"
        else:
            risk_level = "THẤP (An toàn)"
            
        # 5.5. CƠ CHẾ LOGIC NGƯỢC (REVERSE HEURISTICS) - Quét 11 yếu tố
        reasons = []
        if risk_probability >= 40: 
            # Nhóm 1: Lỗi từ phía sinh viên (Cố vấn dễ can thiệp)
            if student.Attendance < 70:
                reasons.append(f"Chuyên cần dưới mức an toàn ({student.Attendance}%)")
            if student.Hours_Studied < 10:
                reasons.append(f"Lượng thời gian tự học rất thấp ({student.Hours_Studied}h/tuần)")
            if student.Previous_Scores < 60:
                reasons.append(f"Mất gốc/Điểm nền tảng các kỳ trước thấp ({student.Previous_Scores}/100)")
            if student.Motivation_Level in ["Thấp", "Rất thấp", "Low"]:
                reasons.append(f"Động lực học tập hiện tại suy giảm đáng kể")
            if student.Sleep_Hours < 6:
                reasons.append(f"Chế độ nghỉ ngơi kém, thể lực suy nhược ({student.Sleep_Hours}h/ngày)")
            if student.Peer_Influence in ["Tiêu cực", "Negative"]:
                reasons.append(f"Chịu ảnh hưởng tiêu cực từ môi trường bạn bè")
            
            # Nhóm 2: Lỗi khách quan / Ngoại cảnh
            if student.Distance_from_Home in ["Xa", "Rất xa", "Far"]:
                reasons.append(f"Khoảng cách di chuyển xa ảnh hưởng đến sức khỏe và thời gian")
            if student.Extracurricular_Activities in ["Nhiều", "Quá mức", "High"]:
                reasons.append(f"Hoạt động ngoại khóa dày đặc, phân tán việc học")
            if student.Access_to_Resources in ["Thấp", "Kém", "Low"]:
                reasons.append(f"Thiếu thốn tài nguyên/thiết bị phục vụ học tập")
            if student.Family_Income in ["Thấp", "Khó khăn", "Low"]:
                reasons.append(f"Có thể đang chịu áp lực về điều kiện tài chính gia đình")
            if student.Teacher_Quality in ["Thấp", "Kém", "Low"]:
                reasons.append(f"Gặp khó khăn trong việc tương thích với phương pháp giảng dạy")
            
            # Chốt chặn cuối cùng nếu không bắt được lỗi nào cụ thể
            if len(reasons) == 0:
                reasons.append("Phát hiện rủi ro tiềm ẩn từ mô hình học máy chưa rõ nguyên nhân cụ thể.")
                
        # 5.6. Trả kết quả JSON về cho Backend/Frontend
        return {
            "status": "success",
            "risk_score_percent": risk_probability,
            "risk_level": risk_level,
            "reasons": reasons,
            "message": "AI đã phân tích dữ liệu thành công"
        }
        
    except Exception as e:
        # Báo lỗi 500 nếu Code Python bị văng lỗi (Giúp BE dễ debug)
        raise HTTPException(status_code=500, detail=f"Lỗi Server AI: {str(e)}")