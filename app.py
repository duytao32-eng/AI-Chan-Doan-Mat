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
# 1. CẤU HÌNH GIAO DIỆN & CSS THÔNG MINH
# ==========================================
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS Tự động tương thích Light/Dark Mode
st.markdown("""
<style>
    /* Hiệu ứng mượt mà khi xuất hiện */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Thẻ kết quả tự động nhận diện màu nền của chế độ Sáng/Tối */
    .result-card {
        background-color: var(--secondary-background-color);
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid;
        animation: fadeSlideUp 0.5s ease-out forwards;
        margin: 10px 0px 25px 0px;
    }
    
    /* Màu viền phân loại bệnh */
    .healthy { border-left-color: #00cc66; } /* Xanh lá */
    .disease { border-left-color: #ff4b4b; } /* Đỏ chuẩn Streamlit */
    
    /* Nhãn độ tin cậy tự động nổi bật */
    .confidence-badge {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 15px;
        border: 1px solid rgba(128,128,128,0.2);
    }
    
    /* Hiệu ứng hover cho ảnh */
    img {
        border-radius: 8px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    img:hover {
        transform: scale(1.02);
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
    st.info("Hệ thống dự đoán đa lớp bệnh lý võng mạc thông qua ảnh nội soi đáy mắt.")
    st.markdown("👨‍🎓 **SVTH:** [Tên của bạn]")
    st.markdown("👨‍🏫 **GVHD:** [Tên Giáo viên]")

# Căn giữa tiêu đề bằng HTML đơn giản không ép màu
st.markdown("<h1 style='text-align: center;'>👁️ AI CHẨN ĐOÁN VÕNG MẠC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; opacity: 0.7;'>Phát hiện tự động 11 loại bệnh lý về mắt với độ chính xác cao.</p>", unsafe_allow_html=True)
st.write("")

# Khu vực Upload căn giữa
col_up1, col_up2, col_up3 = st.columns([1, 4, 1])
with col_up2:
    uploaded_file = st.file_uploader("📥 Tải ảnh đáy mắt lên tại đây (Hỗ trợ: PNG, JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    st.divider()
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Hiển thị ảnh Trước / Sau
    st.subheader("🖼️ Xử lý hình ảnh", anchor=False)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.write("**📷 Ảnh Gốc**")
        st.image(original_image, use_container_width=True)
    with col_img2:
        st.write("**⚙️ Đã Cắt Viền Đen (Crop)**")
        cropped_image = crop_fundus(original_image)
        st.image(cropped_image, use_container_width=True)
        
    st.write("")
    
    # Nút bấm Primary tự động tương thích giao diện
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_button = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner('⏳ AI đang phân tích dữ liệu điểm ảnh...'):
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
        
        # Hiển thị Kết Quả bên ngoài spinner
        if success:
            st.toast('Hoàn tất phân tích!', icon='✅')
            if is_healthy:
                st.balloons()
            
            st.write("")
            st.subheader("📊 Kết Quả Chẩn Đoán", anchor=False)
            
            card_status = "healthy" if is_healthy else "disease"
            top1_name = class_names[top3_idx[0]]
            top1_conf = probs[top3_idx[0]] * 100
            
            # Màu cho TÊN BỆNH (Xanh lá nếu khỏe, Đỏ nếu bệnh)
            color_hex = "#00cc66" if is_healthy else "#ff4b4b"
            
            # Thẻ kết quả thích ứng 100% với Sáng/Tối
            st.markdown(f"""
            <div class='result-card {card_status}'>
                <div style='font-size: 1.1rem; opacity: 0.7; margin-bottom: 5px; text-transform: uppercase;'>Nguy cơ cao nhất:</div>
                <div style='font-size: 2rem; font-weight: 800; color: {color_hex}; line-height: 1.2;'>{top1_name}</div>
                <div class='confidence-badge'>Độ tin cậy: {top1_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write("**Các nguy cơ khác:**")
            
            # In Top 2 và 3 sử dụng hàm gốc của Streamlit
            for i in range(1, 3):
                idx = top3_idx[i]
                disease_name = class_names[idx]
                confidence = probs[idx] * 100
                st.caption(f"{disease_name} (**{confidence:.2f}%**)")
                st.progress(int(confidence))
            
            st.write("")    
            st.info("💡 **Khuyến nghị:** Vui lòng kết hợp với khám lâm sàng chuyên khoa để có kết luận chính xác nhất.")