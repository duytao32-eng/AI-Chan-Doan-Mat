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
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Tổng thể App */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Chỉnh nút bấm siêu to khổng lồ cho Mobile */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2874A6, #1B4F72);
        color: white;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 18px !important;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease 0s;
        width: 100%;
        box-shadow: 0px 6px 15px rgba(40, 116, 166, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1B4F72, #2874A6);
        box-shadow: 0px 8px 20px rgba(40, 116, 166, 0.6);
        transform: translateY(-3px);
    }
    
    /* Khung Expander chứa ảnh (giấu ảnh đi cho gọn) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Animation Thẻ Kết Quả */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-left: 8px solid #2e86c1;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .result-healthy { border-left-color: #2ECC71; }
    .result-disease { border-left-color: #E74C3C; }
    
    .result-card h4 {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-card h2 {
        margin-top: 0;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .result-card .confidence {
        font-size: 1.2rem;
        font-weight: 600;
        color: #34495e;
        background-color: #ecf0f1;
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        margin-top: 10px;
    }
    
    /* Khung Up ảnh */
    .css-1v0mbdj.etr89bj1 {
        border: 2px dashed #3498db;
        border-radius: 15px;
        background-color: #ebf5fb;
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
# 4. GIAO DIỆN CHÍNH (Đã Tối Ưu Cho Mobile)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865744.png", width=100)
    st.markdown("## THÔNG TIN ĐỒ ÁN")
    st.info("Hệ thống dự đoán đa lớp bệnh lý võng mạc thông qua ảnh nội soi đáy mắt.")
    st.markdown("👨‍🎓 **SVTH:** [Tên của bạn]")
    st.markdown("👨‍🏫 **GVHD:** [Tên Giáo viên]")

st.markdown("<h1 style='text-align: center; color: #1B4F72; font-size: 2.2rem;'>👁️ AI CHẨN ĐOÁN VÕNG MẠC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1rem; margin-bottom: 30px;'>Phát hiện tự động 11 loại bệnh lý về mắt với độ chính xác cao.</p>", unsafe_allow_html=True)

# 1. Khu vực Upload (Được căn giữa)
col_up1, col_up2, col_up3 = st.columns([1, 4, 1])
with col_up2:
    uploaded_file = st.file_uploader("Tải ảnh đáy mắt lên (PNG, JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # 2. Giấu ảnh vào Expander để tiết kiệm không gian trên Mobile
    with st.expander("🖼️ Xem ảnh đã tải lên và xử lý", expanded=False):
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Ảnh Gốc</p>", unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
        with col_img2:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Đã Cắt Viền Đen (Crop)</p>", unsafe_allow_html=True)
            cropped_image = crop_fundus(original_image)
            st.image(cropped_image, use_container_width=True)
    
    if 'cropped_image' not in locals():
        cropped_image = crop_fundus(original_image)
        
    st.write("")
    
    # 3. Nút bấm Khổng lồ
    col_btn1, col_btn2, col_btn3 = st.columns([1, 4, 1])
    with col_btn2:
        predict_button = st.button("🚀 Bắt Đầu Chẩn Đoán")
    
    if predict_button:
        with st.spinner('⏳ AI đang phân tích hàng triệu điểm ảnh...'):
            time.sleep(0.8) # Hiệu ứng chờ chân thực
            try:
                model = load_model()
                tensor_img = eval_transform(cropped_image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probs = F.softmax(outputs, dim=1)[0].numpy()
                
                top3_idx = np.argsort(probs)[-3:][::-1]
                is_healthy = (top3_idx[0] == 6)
                success = True
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
                success = False
        
        # 4. Hiển thị Kết Quả (Nổi bật)
        if success:
            st.toast('Hoàn tất phân tích!', icon='✅')
            if is_healthy:
                st.balloons()
            
            # Khối UI Kết quả Top 1
            col_res1, col_res2, col_res3 = st.columns([1, 6, 1])
            with col_res2:
                card_class = "result-healthy" if is_healthy else "result-disease"
                top1_name = class_names[top3_idx[0]]
                top1_conf = probs[top3_idx[0]] * 100
                color_hex = "#2ECC71" if is_healthy else "#E74C3C"
                
                st.markdown(f"""
                <div class='result-card {card_class}'>
                    <h4>Chẩn đoán chính:</h4>
                    <h2 style='color: {color_hex};'>{top1_name}</h2>
                    <div class='confidence'>Độ tin cậy: {top1_conf:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 2 & 3
                st.markdown("<h4 style='color: #34495e;'>📊 CÁC NGUY CƠ KHÁC</h4>", unsafe_allow_html=True)
                for i in range(1, 3):
                    idx = top3_idx[i]
                    disease_name = class_names[idx]
                    confidence = probs[idx] * 100
                    st.write(f"{disease_name} (**{confidence:.1f}%**)")
                    st.progress(int(confidence))
                
                st.info("💡 Nên kết hợp với khám lâm sàng để có kết luận chính xác nhất.")