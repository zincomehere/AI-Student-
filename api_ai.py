# ==============================================================================
# LỆNH KHỞI ĐỘNG SERVER LOCAL: uvicorn api_ai:app --reload
# ==============================================================================

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import json
import traceback

app = FastAPI(title="Hệ thống Cảnh báo Rủi ro Sinh viên - Ultimate AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến toàn cục lưu trữ mô hình
model = None
encoders = None
surrogate_tree = None
cay_text = ""
bang_quy_doi = {}

FEATURES = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
    'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality'
]

FEATURE_IMPORTANCE_ORDER = [
    'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
    'Motivation_Level', 'Peer_Influence', 'Sleep_Hours', 'Family_Income', 
    'Distance_from_Home', 'Teacher_Quality', 'Extracurricular_Activities'
]

# 1. BỌC ÁO GIÁP PYDANTIC (Đã thêm cờ nhận diện thang điểm)
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
        extra = "ignore" 

@app.on_event("startup")
def load_ai_assets():
    global model, encoders, surrogate_tree, cay_text, bang_quy_doi
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        encoders = joblib.load("label_encoders.pkl")
        
        surrogate_tree = joblib.load("surrogate_tree.pkl")
        with open("cay_tong_quat.txt", "r", encoding="utf-8") as f:
            cay_text = [line for line in f.read().split('\n') if line.strip()]
            
        for col, encoder in encoders.items():
            bang_quy_doi[str(col)] = {str(label): int(val) for label, val in zip(encoder.classes_, range(len(encoder.classes_)))}
            
        print("✅ Nạp thành công Toàn bộ Mô hình CatBoost và Cây Quyết Định!")
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG (Thiếu file): {e}")

# ==============================================================================
# HÀM BỔ TRỢ
# ==============================================================================
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if pd.isna(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (str, bool, type(None))):
        return obj
    else:
        return str(obj)

# 🔥 ĐÃ FIX: HÀM NÀY NHẬN THÊM BIẾN `catboost_risk_level` ĐỂ ÉP CUNG
def trich_xuat_duong_di_mot_sv(X_encoded_row, catboost_risk_level):
    if surrogate_tree is None:
        return ["Lỗi: Hệ thống chưa nạp được Cây Quyết Định"]

    if X_encoded_row.ndim == 1:
        X_encoded_row = X_encoded_row.reshape(1, -1)
        
    node_indicator = surrogate_tree.decision_path(X_encoded_row)
    feature_idx = surrogate_tree.tree_.feature
    threshold = surrogate_tree.tree_.threshold
    
    lo_trinh = []
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    for node_id in node_index:
        if feature_idx[node_id] == -2: break
            
        ten_bien = str(FEATURES[int(feature_idx[node_id])])
        gia_tri_nguong = float(round(threshold[node_id], 2))
        gia_tri_sv = float(round(X_encoded_row[0, int(feature_idx[node_id])], 2))
        
        huong_di = f"<= {gia_tri_nguong} (Rẽ TRÁI)" if gia_tri_sv <= gia_tri_nguong else f"> {gia_tri_nguong} (Rẽ PHẢI)"
        lo_trinh.append(f"Xét [{ten_bien}]: Đạt {gia_tri_sv} {huong_di}")
        
    # 🔥 ĐÃ FIX LỖI "CHÓ ĐÁ MÈO": Dùng kết quả CatBoost chốt câu cuối cùng
    if "Nguy hiểm" in catboost_risk_level or "CAO" in catboost_risk_level:
        lo_trinh.append(f"==> DỰ ĐOÁN CUỐI CÙNG TỪ LÕI AI: {catboost_risk_level} ")
    elif "Theo dõi" in catboost_risk_level or "TRUNG BÌNH" in catboost_risk_level:
        lo_trinh.append(f"==> DỰ ĐOÁN CUỐI CÙNG TỪ LÕI AI: {catboost_risk_level} ")
    else:
        lo_trinh.append(f"==> DỰ ĐOÁN CUỐI CÙNG TỪ LÕI AI: {catboost_risk_level} ")
        
    return lo_trinh

def extract_sorted_reasons(row_dict, risk_percent):
    reasons = []
    if risk_percent < 40:
        return ["Sinh viên đang duy trì các chỉ số học tập và sinh hoạt ở mức an toàn."]
        
    for feature in FEATURE_IMPORTANCE_ORDER:
        val = row_dict.get(feature)
        if val is None: continue
        
        if feature == 'Attendance' and float(val) <= 71.5: 
            reasons.append(f"Tỷ lệ chuyên cần thấp ({val}%)")
        elif feature == 'Hours_Studied' and float(val) <= 14.5:
            reasons.append(f"Thời gian tự học quá ít ({val} giờ/tuần)")
        elif feature == 'Previous_Scores' and float(val) <= 65.5:
            reasons.append(f"Điểm nền tảng kỳ trước yếu ({val}/100)")
        elif feature == 'Access_to_Resources' and str(val) == 'Low':
            reasons.append(f"Thiếu thốn tài nguyên phục vụ học tập")
        elif feature == 'Motivation_Level' and str(val) == 'Low':
            reasons.append(f"Động lực học tập đang ở mức Thấp")
        elif feature == 'Peer_Influence' and str(val) == 'Negative':
            reasons.append(f"Chịu ảnh hưởng tiêu cực từ bạn bè")
        elif feature == 'Sleep_Hours' and float(val) <= 6.0:
            reasons.append(f"Thiếu ngủ, thể trạng kém ({val} giờ/đêm)")
        elif feature == 'Family_Income' and str(val) == 'Low':
            reasons.append(f"Đang gặp áp lực về tài chính gia đình")
        elif feature == 'Distance_from_Home' and str(val) == 'Far':
            reasons.append(f"Di chuyển quá xa, mất nhiều thời gian")
        elif feature == 'Teacher_Quality' and str(val) == 'Low':
            reasons.append(f"Chưa thích nghi với chất lượng/phương pháp giảng dạy")
            
    if not reasons:
        reasons.append("Rủi ro tổng hợp từ sự giao thoa phức tạp của nhiều yếu tố nhỏ.")
        
    return reasons

# ==============================================================================
# LÕI XỬ LÝ CHÍNH (ĐÃ THÊM ĐIỂM LIỆT + AN TOÀN QUY ĐỔI ĐIỂM)
# ==============================================================================
def process_batch_logic(students: List[StudentData]):
    if model is None or surrogate_tree is None:
        return {"error": "Server chưa nạp được mô hình. Vui lòng kiểm tra lại thư mục chạy."}

    data_dicts = [student.model_dump() if hasattr(student, 'model_dump') else student.dict() for student in students]
    
    # 1. QUY ĐỔI ĐIỂM AN TOÀN
    for row in data_dicts:
        diem = row['Previous_Scores']
        loai_thang_diem = row.get('Scale_Type', '100')
        
        # Chỉ nhân nếu Frontend chỉ định rõ thang điểm 4 hoặc 10, để tránh nhân nhầm điểm 8/100
        if loai_thang_diem == '4' or (loai_thang_diem == '100' and diem <= 4.0): 
            row['Previous_Scores'] = float(diem * 25)
        elif loai_thang_diem == '10' or (loai_thang_diem == '100' and 4.0 < diem <= 10.0): 
            row['Previous_Scores'] = float(diem * 10)
            
    df_input = pd.DataFrame(data_dicts)
    df_encoded = df_input.copy()
    
    # 2. LABEL ENCODING
    for col in encoders:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            mapping_dict = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
            df_encoded[col] = df_encoded[col].map(mapping_dict).fillna(0).astype(int)
            
    X_encoded = df_encoded[FEATURES]
    
    # DỰ BÁO TOÀN BỘ BẰNG CATBOOST
    risk_probabilities = model.predict_proba(X_encoded)[:, 1] * 100
    
    batch_results = []
    
    # 3. KẾT HỢP ĐIỂM LIỆT + KẾT QUẢ AI
    for i, risk in enumerate(risk_probabilities):
        sv_dict = data_dicts[i]
        
        # 🔥 TẦNG 1: LƯỚI LỌC "ĐIỂM LIỆT" (QUY CHẾ TRƯỜNG)
        if sv_dict['Attendance'] < 70.0:
            batch_results.append({
                "index": int(i), 
                "risk_score_percent": 100.0,
                # TRẢ LẠI ĐÚNG CHỮ FRONTEND CẦN ĐỂ KHÔNG GÃY UI
                "risk_level": "CAO (Nguy hiểm)", 
                # NHÉT CHỮ CẤM THI VÀO ĐÂY THEO ĐÚNG Ý FRONTEND
                "sorted_reasons_for_ui": [f"BÁO ĐỘNG ĐỎ: Vắng mặt quá 30% (Đạt {sv_dict['Attendance']}%) - MỨC CẤM THI THEO QUY CHẾ."],
                "ai_explanation_path": ["Hệ thống phát hiện vi phạm Quy chế cứng về Chuyên cần.", "==> XỬ LÝ KHẨN CẤP: ĐÌNH CHỈ / CẤM THI "],
                "original_features": sv_dict
            })
            continue 
            
        elif sv_dict['Previous_Scores'] <= 0.0:
            batch_results.append({
                "index": int(i), 
                "risk_score_percent": 100.0,
                "risk_level": "CAO (Nguy hiểm)", # TRẢ LẠI ĐÚNG CHỮ
                "sorted_reasons_for_ui": ["BÁO ĐỘNG ĐỎ: Điểm tích lũy bằng 0 - MỨC ĐÌNH CHỈ HỌC TẬP."],
                "ai_explanation_path": ["Vi phạm ranh giới điểm số cốt lõi.", "==> XỬ LÝ KHẨN CẤP: ĐÌNH CHỈ / CẤM THI "],
                "original_features": sv_dict
            })
            continue

        # 🔥 TẦNG 2: XỬ LÝ KẾT QUẢ AI (CHO NHỮNG CA QUA ĐƯỢC ĐIỂM LIỆT) 🔥
        risk_percent = float(round(float(risk), 2))
        
        if risk_percent >= 65: risk_level = "CAO (Nguy hiểm)"
        elif risk_percent >= 40: risk_level = "TRUNG BÌNH (Cần theo dõi)"
        else: risk_level = "THẤP (An toàn)"
        
        X_row_array = X_encoded.iloc[i].values
        
        # Truyền thêm risk_level vào hàm dò đường
        lo_trinh_ai = trich_xuat_duong_di_mot_sv(X_row_array, risk_level)
        sorted_reasons = extract_sorted_reasons(sv_dict, risk_percent)

        batch_results.append({
            "index": int(i), 
            "risk_score_percent": risk_percent,
            "risk_level": str(risk_level),
            "sorted_reasons_for_ui": sorted_reasons,
            "ai_explanation_path": lo_trinh_ai, 
            "original_features": sv_dict
        })
        
    return {
        "status": "success",
        "total_processed": int(len(students)),
        "tree_rules_for_professor": cay_text,
        "label_encoding_map": bang_quy_doi,
        "results": batch_results
    }

# ==============================================================================
# API QUÉT SỈ (BATCH) VÀ QUÉT LẺ (SINGLE)
# ==============================================================================
@app.post("/api/predict_batch")
def predict_risk_batch(students: List[StudentData]):
    try:
        raw_response = process_batch_logic(students)
        if "error" in raw_response:
            return Response(content=json.dumps({"detail": raw_response["error"]}, ensure_ascii=False), status_code=503, media_type="application/json")
            
        cleaned_response = clean_for_json(raw_response)
        json_str = json.dumps(cleaned_response, ensure_ascii=False)
        return Response(content=json_str, media_type="application/json")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return Response(
            content=json.dumps({"detail": f"Lỗi nội bộ Python: {str(e)}", "traceback_de_bug": error_trace}, ensure_ascii=False),
            status_code=500,
            media_type="application/json"
        )

@app.post("/api/predict")
def predict_risk_single(student: StudentData):
    try:
        raw_response = process_batch_logic([student])
        if "error" in raw_response:
            return Response(content=json.dumps({"detail": raw_response["error"]}, ensure_ascii=False), status_code=503, media_type="application/json")
            
        single_response = {
            "status": "success",
            "tree_rules_for_professor": raw_response["tree_rules_for_professor"],
            "label_encoding_map": raw_response["label_encoding_map"],
            "result": raw_response["results"][0]
        }
        
        cleaned_response = clean_for_json(single_response)
        json_str = json.dumps(cleaned_response, ensure_ascii=False)
        return Response(content=json_str, media_type="application/json")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return Response(
            content=json.dumps({"detail": f"Lỗi nội bộ Python: {str(e)}", "traceback_de_bug": error_trace}, ensure_ascii=False),
            status_code=500,
            media_type="application/json"
        )

@app.get("/")
def health_check():
    trang_thai_mo_hinh = "SẴN SÀNG" if (model is not None and surrogate_tree is not None) else "LỖI THIẾU FILE!"
    return Response(content=json.dumps({"status": "ok", "message": "API Explainable AI (Hybrid Rule-Based) đang hoạt động!", "model_status": trang_thai_mo_hinh}, ensure_ascii=False), media_type="application/json")