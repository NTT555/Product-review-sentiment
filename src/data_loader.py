import pandas as pd
import os

def load_data(filepath):
    print(f"⏳ Đang tải dữ liệu từ: {filepath}...")
    
    # Tự động kiểm tra đuôi file để áp dụng hàm đọc chính xác
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.xlsx':
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError("❌ Định dạng file không hỗ trợ! Vui lòng sử dụng file .xlsx hoặc .csv")
    
    # Kiểm tra các cột bắt buộc phải có
    if 'Text' not in df.columns or 'label' not in df.columns:
        raise KeyError("❌ Cấu trúc dữ liệu sai! Bắt buộc phải có cột 'Text' và 'label'.")
        
    # Loại bỏ các dòng trống không có nội dung văn bản hoặc chưa được điền nhãn
    df = df.dropna(subset=['Text', 'label'])
    
    # Chuẩn hóa nhãn: Xóa khoảng trắng thừa và chuyển về chữ thường (tránh lệch nhãn 'Positive' và 'positive')
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    
    # Giữ lại các dòng có nhãn hợp lệ (loại bỏ các dòng trung lập hoặc dữ liệu nhiễu nếu có)
    df = df[df['label'].isin(['positive', 'negative'])]
    
    print(f"✅ Tải dữ liệu hoàn tất! Thu thập được {len(df)}.")
    return df