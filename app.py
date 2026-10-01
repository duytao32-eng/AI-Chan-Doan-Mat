import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & CSS ANIMATION TÙY CHỈNH
# ==========================================
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide")

# CSS Animation mượt mà
st.markdown("""
<style>
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card {
        background-color: #f8f9f9;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-left: 6px solid #2e86c1;
        animation: slideUpFade 0.6s ease-out forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .result-healthy { border-left-color: #28b463; }
    .result-disease { border-left-color: #e74c3c; }
    
    img {
        border-radius: 10px;
        transition: transform 0.4s ease;
    }
    img:hover {
        transform: scale(1.02);
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1f77b4, #154360);
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease 0s;
        width: 100%;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #154360, #1f77b4);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KHAI BÁO KIẾN TRÚC MÔ HÌNH EUPE-ViT
# ==========================================
class OptimalAttentiveProbe(nn.Module):
    def __init__(self, in_dim=384, num_classes=11, dropout_p=0.1): 
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(in_dim, num_classes)
        
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        attended_features = x * attn_weights
        features_dropped = self.dropout(attended_features) 
        return self.classifier(features_dropped)

class EUPE_ViT_Optimal(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        self.head = OptimalAttentiveProbe(in_dim=384, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ==========================================
# 3. HÀM XỬ LÝ (CACHE ĐỂ TĂNG TỐC)
# ==========================================
@st.cache_resource
def load_model():
    model = EUPE_ViT_Optimal(num_classes=11)
    device = torch.device('cpu') 
    model.load_state_dict(torch.load('eupe_vits16_best_stage2.pth', map_location=device))
    model.eval()
    return model

def crop_fundus(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img_cropped = img_bgr[y:y+h, x:x+w]
    else:
        img_cropped = img_bgr
    return Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

class_names = [
    'Advanced/End-stage Glaucoma',
    'Dry Age-Related Macular Degeneration',
    'Mild Diabetic Retinopathy',
    'Mild Glaucoma',
    'Moderate Diabetic Retinopathy',
    'Moderate Glaucoma',
    'No Age-Related Macular Degeneration (Khỏe mạnh)',
    'Proliferative Diabetic Retinopathy (PDR)',
    'Severe Diabetic Retinopathy',
    'Severe Glaucoma',
    'Wet Age-Related Macular Degeneration'
]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. TRANG CHÍNH & SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865744.png", width=80)
    st.markdown("## THÔNG TIN ĐỒ ÁN")
    st.info("Đề tài: Dự đoán đa lớp bệnh lý võng mạc thông qua ảnh nội soi đáy mắt sử dụng Học Sâu.")
    
    st.markdown("### Nhóm Thực Hiện")
    st.markdown("👨‍🎓 **SVTH:** [Tên của bạn]")
    st.markdown("👨‍🏫 **GVHD:** [Tên Giáo viên]")
    
    st.divider()
    st.markdown("### Công nghệ")
    st.markdown("🧠 **AI Model:** Gated EUPE (DINO ViT-S/16)")
    st.markdown("⚙ **FrameWork:** PyTorch, OpenCV, Streamlit")

st.markdown("<h1 style='text-align: center; color: #154360;'>👁️ HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN VÕNG MẠC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #5d6d7e;'>Phân tích và phát hiện tự động 11 loại bệnh lý về mắt với độ chính xác cao.</p>", unsafe_allow_html=True)
st.write("")

# Khu vực Upload
upload_col, _ = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader("📥 Tải ảnh đáy mắt lên tại đây (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.divider()
    original_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        st.markdown("<h4 style='text-align: center; color: #34495e;'>📷 Ảnh Gốc</h4>", unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.markdown("<h4 style='text-align: center; color: #34495e;'>⚙️ Ảnh Crop Tự Động</h4>", unsafe_allow_html=True)
        cropped_image = crop_fundus(original_image)
        st.image(cropped_image, use_container_width=True)
        
    with col3:
        st.markdown("<h4 style='text-align: center; color: #34495e;'>📊 Kết Quả Chẩn Đoán</h4>", unsafe_allow_html=True)
        predict_button = st.button("🚀 XỬ LÝ PHÂN TÍCH")
        
        if predict_button:
            # 1. Chạy tiến trình phân tích ngầm
            with st.spinner('⏳ AI đang quét và phân tích tổn thương...'):
                try:
                    model = load_model()
                    tensor_img = eval_transform(cropped_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = model(tensor_img)
                        probs = F.softmax(outputs, dim=1)[0].numpy()
                    
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    is_healthy = (top3_idx[0] == 6)
                    success = True
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
                    success = False
            
            # 2. Hiển thị kết quả sau khi phân tích xong (Đưa ra ngoài st.spinner)
            if success:
                st.toast('Hoàn tất phân tích dữ liệu!', icon='✅')
                if is_healthy:
                    st.balloons()
                
                card_class = "result-healthy" if is_healthy else "result-disease"
                top1_name = class_names[top3_idx[0]]
                top1_conf = probs[top3_idx[0]] * 100
                
                # Sửa lỗi chữ ẩn trong Dark Mode bằng cách thêm color: #2c3e50
                st.markdown(f"""
                <div class='result-card {card_class}'>
                    <h4 style='margin-top: 0; color: #2c3e50;'>Nguy cơ cao nhất:</h4>
                    <h3 style='color: {"#28b463" if is_healthy else "#e74c3c"}; margin-bottom: 0;'>{top1_name}</h3>
                    <p style='font-size: 18px; font-weight: bold; color: #2c3e50;'>Độ tin cậy: {top1_conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                
                st.markdown("**Các nguy cơ khác:**")
                for i in range(1, 3):
                    idx = top3_idx[i]
                    disease_name = class_names[idx]
                    confidence = probs[idx] * 100
                    st.write(f"{disease_name}: **{confidence:.2f}%**")
                    st.progress(int(confidence))