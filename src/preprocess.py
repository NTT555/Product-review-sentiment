import re
from underthesea import word_tokenize

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Xóa URL
    text = re.sub(r'[^\w\s]', '', text) # Xóa dấu câu
    text = re.sub(r'\d+', '', text) # Xóa số
    
    # Tách từ tiếng Việt (nối bằng dấu '_')
    text = word_tokenize(text, format="text")
    
    return text.strip()