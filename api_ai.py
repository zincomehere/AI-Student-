# ==============================================================================
# LỆNH KHỞI ĐỘNG SERVER LOCAL: uvicorn api_ai:app --reload
# ==============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import joblib
from catboost import CatBoostClassifier

app = FastAPI(title="Hệ thống Cảnh báo Rủi ro Sinh viên - AI API", version="Final_Master")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
encoders = None

FEATURES = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
    'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality'
]

# 1. BỌC ÁO GIÁP PYDANTIC & CHO PHÉP DỮ LIỆU THỪA TỪ CSV
class StudentData(BaseModel):
    Attendance: float = Field(default=75.0)
    Hours_Studied: float = Field(default=10.0)
    Previous_Scores: float = Field(default=65.0)
    Access_to_Resources: str = Field(default="Medium")
    Motivation_Level: str = Field(default="Medium")
    Family_Income: str = Field(default="Medium")
    Peer_Influence: str = Field(default="Neutral")
    Distance_from_Home: str = Field(default="Moderate")
    Extracurricular_Activities: str = Field(default="No")
    Sleep_Hours: float = Field(default=7.0)
    Teacher_Quality: str = Field(default="Medium")

    class Config:
        extra = "ignore" # BÍ KÍP: Backend ném 20 cột sang cũng không báo lỗi, AI chỉ bốc 11 cột này ra xài.

@app.on_event("startup")
def load_ai_assets():
    global model, encoders
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        encoders = joblib.load("label_encoders.pkl")
        print("✅ Nạp thành công Mô hình & Từ điển!")
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Server AI đang hoạt động rất khỏe!"}

# ==============================================================================
# API XỬ LÝ SỈ DÀNH CHO BẢNG DỮ LIỆU TỪ WEB
# ==============================================================================
@app.post("/api/predict_batch")
def predict_risk_batch(students: List[StudentData]):
    # Chống làm treo Server
    if len(students) > 5000:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tối đa 5000 sinh viên/lần quét.")

    try:
        data_dicts = [student.dict() for student in students]
        
        # Sửa lỗi khác biệt hệ điểm (Nếu điểm hệ 10 thì nhân 10)
        for row in data_dicts:
            if row['Previous_Scores'] <= 10.0:
                row['Previous_Scores'] = row['Previous_Scores'] * 10
                
        df_input = pd.DataFrame(data_dicts)
        
        # Tiền xử lý: Dịch Label Encoding
        for col in encoders:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
                mapping_dict = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
                df_input[col] = df_input[col].map(mapping_dict).fillna(0).astype(int)
                
        X_input = df_input[FEATURES]
        risk_probabilities = model.predict_proba(X_input)[:, 1] * 100
        
        batch_results = []
        for i, risk in enumerate(risk_probabilities):
            student_info = students[i]
            
            risk_percent = round(risk, 2)
            if risk_percent >= 65: risk_level = "CAO (Nguy hiểm)"
            elif risk_percent >= 40: risk_level = "TRUNG BÌNH (Cần theo dõi)"
            else: risk_level = "THẤP (An toàn)"
                
            reasons = []
            if risk_percent >= 40:
                # 1. Bắt lỗi Số (Cộng thêm logic xử lý điểm hệ 10 cho đúng logic cảnh báo)
                diem_kiem_tra = student_info.Previous_Scores * 10 if student_info.Previous_Scores <= 10 else student_info.Previous_Scores
                
                if student_info.Attendance < 70: reasons.append(f"Chuyên cần thấp ({student_info.Attendance}%)")
                if diem_kiem_tra < 60: reasons.append(f"Mất gốc/Điểm cũ thấp ({diem_kiem_tra}/100)")
                if student_info.Hours_Studied < 10: reasons.append(f"Tự học rất ít ({student_info.Hours_Studied}h/tuần)")
                if student_info.Sleep_Hours < 6: reasons.append(f"Thiếu ngủ, thể trạng kém ({student_info.Sleep_Hours}h/ngày)")
                
                # 2. Bắt lỗi chữ (Đã match 100% với CSV tiếng Anh gốc)
                if student_info.Motivation_Level in ["Low"]: reasons.append("Động lực học tập hiện tại suy giảm đáng kể")
                if student_info.Family_Income in ["Low"]: reasons.append("Có áp lực về điều kiện tài chính gia đình")
                if student_info.Distance_from_Home in ["Far"]: reasons.append("Khoảng cách di chuyển quá xa ảnh hưởng sức khỏe")
                if student_info.Access_to_Resources in ["Low"]: reasons.append("Thiếu thốn tài nguyên/thiết bị phục vụ học tập")
                if student_info.Peer_Influence in ["Negative"]: reasons.append("Chịu ảnh hưởng tiêu cực từ môi trường bạn bè")
                if student_info.Teacher_Quality in ["Low"]: reasons.append("Chưa tương thích với phương pháp giảng dạy")

                if len(reasons) == 0: reasons.append("Có rủi ro tiềm ẩn (AI đánh giá tổng hợp đa biến)")

            batch_results.append({
                "index": i, 
                "risk_score_percent": risk_percent,
                "risk_level": risk_level,
                "reasons": reasons,
                "original_features": student_info.dict() 
            })
            
        return {
            "status": "success",
            "total_processed": len(students),
            "results": batch_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server AI: {str(e)}")

@app.post("/api/predict")
def predict_risk(student: StudentData):
    batch_result = predict_risk_batch([student])
    return batch_result["results"][0]