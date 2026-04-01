# ==============================================================================
# LỆNH KHỞI ĐỘNG SERVER LOCAL: uvicorn api_ai:app --reload
# ==============================================================================

from fastapi import FastAPI, Response, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import joblib
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_text
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import json
import traceback
from datetime import datetime

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

# 1. BỌC ÁO GIÁP PYDANTIC
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
# LÕI XỬ LÝ CHÍNH
# ==============================================================================
def process_batch_logic(students: List[StudentData]):
    if model is None or surrogate_tree is None:
        return {"error": "Server chưa nạp được mô hình. Vui lòng kiểm tra lại thư mục chạy."}

    data_dicts = [student.model_dump() if hasattr(student, 'model_dump') else student.dict() for student in students]
    
    # QUY ĐỔI ĐIỂM AN TOÀN
    for row in data_dicts:
        diem = row['Previous_Scores']
        loai_thang_diem = row.get('Scale_Type', '100')
        
        if loai_thang_diem == '4' or (loai_thang_diem == '100' and diem <= 4.0): 
            row['Previous_Scores'] = float(diem * 25)
        elif loai_thang_diem == '10' or (loai_thang_diem == '100' and 4.0 < diem <= 10.0): 
            row['Previous_Scores'] = float(diem * 10)
            
    df_input = pd.DataFrame(data_dicts)
    df_encoded = df_input.copy()
    
    # LABEL ENCODING
    for col in encoders:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            mapping_dict = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
            df_encoded[col] = df_encoded[col].map(mapping_dict).fillna(0).astype(int)
            
    X_encoded = df_encoded[FEATURES]
    
    # DỰ BÁO TOÀN BỘ BẰNG CATBOOST
    risk_probabilities = model.predict_proba(X_encoded)[:, 1] * 100
    
    batch_results = []
    
    # XỬ LÝ KẾT QUẢ TỪ AI (Đã bỏ lưới lọc điểm liệt)
    for i, risk in enumerate(risk_probabilities):
        sv_dict = data_dicts[i]
        
        risk_percent = float(round(float(risk), 2))
        
        if risk_percent >= 65: risk_level = "CAO (Nguy hiểm)"
        elif risk_percent >= 40: risk_level = "TRUNG BÌNH (Cần theo dõi)"
        else: risk_level = "THẤP (An toàn)"
        
        X_row_array = X_encoded.iloc[i].values
        
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
# LÕI TỰ ĐỘNG HỌC LẠI (AUTO-RETRAIN) & ĐỒNG BỘ X-QUANG AI
# ==============================================================================
def tien_hanh_tu_hoc_tu_file_moi(file_path_moi: str):
    """Hàm chạy ngầm xử lý file do Phòng Đào tạo vừa up lên"""
    global model, encoders, bang_quy_doi, surrogate_tree, cay_text 
    
    print("[END-OF-SEMESTER RETRAIN] BẮT ĐẦU TIẾN TRÌNH CẬP NHẬT KIẾN THỨC MỚI...")
    
    try:
        # 1. Đọc file gốc và file mới
        df_goc = pd.read_csv('StudentPerformanceFactors.csv')
        df_moi = pd.read_csv(file_path_moi)
        
        if 'Exam_Score' not in df_moi.columns:
            print("[TỪ CHỐI] File upload không có cột 'Exam_Score'. Bắt buộc phải có điểm thi để AI học lại!")
            return
            
        # Gộp dữ liệu
        df_tong = pd.concat([df_goc, df_moi], ignore_index=True)
        print(f"Đã gộp dữ liệu thành công. Tổng sinh viên: {len(df_tong)}")
        
        # 🔥 FIX LỖI TỬ HUYỆT: LƯU FILE TÍCH LŨY KHI RAW DATA CÒN NGUYÊN VẸN CỘT EXAM_SCORE
        df_tong.to_csv('StudentPerformanceFactors.csv', index=False)
        print(f"Đã lưu tích lũy {len(df_tong)} sinh viên vào kho dữ liệu gốc an toàn!")
            
        cot_giu_lai = [
            'Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 
            'Motivation_Level', 'Family_Income', 'Peer_Influence', 'Distance_from_Home', 
            'Extracurricular_Activities', 'Sleep_Hours', 'Teacher_Quality', 'Exam_Score'
        ]
        
        # 2. Xử lý & Cắt chia
        df_tong = df_tong[cot_giu_lai]
        # Sau dòng này là cột Exam_Score bị biến mất, nên ta phải lưu nó ở phía trên!
        df_tong['Rui_ro'] = df_tong['Exam_Score'].apply(lambda x: 1 if x < 65 else 0)
        df_tong = df_tong.drop('Exam_Score', axis=1)

        new_encoders_dict = {} 
        for col in df_tong.select_dtypes(exclude=['number']).columns:
            le = LabelEncoder()
            df_tong[col] = le.fit_transform(df_tong[col].astype(str))
            new_encoders_dict[col] = le 

        X = df_tong.drop('Rui_ro', axis=1)
        y = df_tong['Rui_ro']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # 3. Tính AUC Não Cũ
        try:
            y_probs_old = model.predict_proba(X_test)[:, 1]
            fpr_old, tpr_old, _ = roc_curve(y_test, y_probs_old)
            auc_old = auc(fpr_old, tpr_old)
        except:
            auc_old = 0.0

        # 4. Huấn luyện Não Mới
        new_model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, loss_function='Logloss', verbose=0)
        new_model.fit(X_train_smote, y_train_smote)
        
        y_probs_new = new_model.predict_proba(X_test)[:, 1]
        fpr_new, tpr_new, _ = roc_curve(y_test, y_probs_new)
        auc_new = auc(fpr_new, tpr_new)

        # 5. CHỐT CHẶN AN TOÀN & HOT-RELOAD
        if auc_new >= 0.80:
            print(f"[RETRAIN] NÃO MỚI TỐT HƠN ({auc_new:.4f} >= 0.80). ĐANG THAY NÃO...")
            
            if os.path.exists("catboost_model.cbm"):
                shutil.copy("catboost_model.cbm", "catboost_model_backup.cbm")
            new_model.save_model("catboost_model.cbm")
            joblib.dump(new_encoders_dict, "label_encoders.pkl")
            
            # Cập nhật Surrogate Tree
            print("Đang đồng bộ lại Cây X-Quang...")
            catboost_predictions = new_model.predict(X)
            new_surrogate = DecisionTreeClassifier(max_depth=7, random_state=42)
            new_surrogate.fit(X, catboost_predictions)
            joblib.dump(new_surrogate, "surrogate_tree.pkl")
            
            new_tree_rules = export_text(new_surrogate, feature_names=FEATURES)
            new_tree_rules = new_tree_rules.replace("class: 0", "Dự báo: AN TOÀN").replace("class: 1", "Dự báo: RỦI RO CAO")
            with open("cay_tong_quat.txt", "w", encoding="utf-8") as f:
                f.write(new_tree_rules)
            
            # Thay não trực tiếp trên RAM
            model = new_model
            encoders = new_encoders_dict
            surrogate_tree = new_surrogate
            cay_text = [line for line in new_tree_rules.split('\n') if line.strip()]
            
            bang_quy_doi.clear()
            for col, encoder in encoders.items():
                bang_quy_doi[str(col)] = {str(label): int(val) for label, val in zip(encoder.classes_, range(len(encoder.classes_)))}
                
            print("🎉 [RETRAIN] HOÀN TẤT THAY NÃO ZERO-DOWNTIME CHUẨN MLOPS!")
        else:
            print(f"[RETRAIN] NÃO MỚI DƯỚI CHUẨN ({auc_new:.4f} < 0.80). TỪ CHỐI CẬP NHẬT!")
            
        # Dọn rác
        if os.path.exists(file_path_moi):
            os.remove(file_path_moi)
            
    except Exception as e:
        print(f"[RETRAIN] LỖI HỆ THỐNG TRONG LÚC HỌC: {e}")
        if os.path.exists(file_path_moi):
            os.remove(file_path_moi)

@app.post("/api/retrain_end_of_semester")
async def api_upload_and_retrain(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ định dạng file CSV.")
        
    temp_file_path = f"temp_retrain_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        background_tasks.add_task(tien_hanh_tu_hoc_tu_file_moi, temp_file_path)
        
        return {
            "status": "success",
            "message": "Đã nhận file điểm học kỳ mới. Hệ thống AI đang tự học ngầm và đồng bộ cây X-Quang. Dự kiến hoàn tất trong 2-3 phút."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {str(e)}")

# ... (Giữ nguyên các khối API Batch, Single và Health Check phía dưới) ...

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
    thoi_gian_update = "Chưa rõ"
    if os.path.exists("catboost_model.cbm"):
        timestamp = os.path.getmtime("catboost_model.cbm")
        thoi_gian_update = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    trang_thai_mo_hinh = "SẴN SÀNG" if (model is not None and surrogate_tree is not None) else "LỖI THIẾU FILE!"
    
    return Response(
        content=json.dumps({
            "status": "ok", 
            "message": "API Explainable AI đang hoạt động!", 
            "model_status": trang_thai_mo_hinh,
            "last_retrain_time": thoi_gian_update
        }, ensure_ascii=False), 
        media_type="application/json"
    )
import os

# API MỚI DÀNH RIÊNG CHO FRONTEND LẤY CÂY X-QUANG HIỂN THỊ
@app.get("/api/model/current_tree")
async def get_current_ai_tree():
    tree_path = "cay_tong_quat.txt" # Tên file chứa cây X-Quang của ông
    
    if not os.path.exists(tree_path):
        return {
            "status": "error", 
            "message": "Chưa có dữ liệu cây. Vui lòng train mô hình trước!"
        }
        
    try:
        with open(tree_path, "r", encoding="utf-8") as f:
            cay_text = [line.strip() for line in f.readlines() if line.strip()]
            
        return {
            "status": "success",
            "message": "✅ Đã lấy thông tin cây Quyết định (Não AI) mới nhất!",
            "data": {
                "version": "Latest_Model",
                "tree_rules": cay_text
            }
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Lỗi đọc file cây: {str(e)}")