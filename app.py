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
# 1. CẤU HÌNH GIAO DIỆN & QUẢN LÝ TRẠNG THÁI (ROUTING)
# ==========================================
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

# Khởi tạo biến trạng thái để chuyển trang
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def change_page(page_name):
    st.session_state.current_page = page_name

# ==========================================
# 2. CSS DYNAMIC TÙY THEO TRANG
# ==========================================
if st.session_state.current_page == 'home':
    # CSS CHO TRANG CHỦ (Nền ảnh Bác sĩ, nút bự ở giữa)
    st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-photo/medical-technology-concept-with-doctor-touching-virtual-screen_53876-104054.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        /* Phủ lớp kính tối nhẹ để chữ nổi bật */
        .block-container {
            background-color: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border-radius: 20px;
            padding: 3rem;
            margin-top: 10vh;
            max-width: 800px;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }
        h1, p { color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
        
        /* Nút bấm chuyển trang siêu mượt */
        div.stButton > button {
            background: linear-gradient(135deg, #1f77b4, #2874A6);
            color: white;
            border-radius: 50px;
            padding: 15px 40px;
            font-size: 20px !important;
            font-weight: bold;
            border: none;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4);
            margin-top: 20px;
        }
        div.stButton > button:hover {
            transform: scale(1.05) translateY(-3px);
            box-shadow: 0 12px 25px rgba(31, 119, 180, 0.6);
            color: white;
        }
        [data-testid="collapsedControl"] { display: none; } /* Ẩn sidebar ở trang chủ */
    </style>
    """, unsafe_allow_html=True)

else:
    # CSS CHO TRANG CHẨN ĐOÁN (Phong cách Dashboard hiện đại giống ảnh 2)
    st.markdown("""
    <style>
        .stApp {
            background-color: var(--background-color);
            background-image: none;
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        /* Top Navigation Bar Ảo */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(128,128,128,0.1);
            margin-bottom: 25px;
        }
        .nav-item {
            font-weight: 600;
            color: #7f8c8d;
            padding: 5px 15px;
            border-bottom: 3px solid transparent;
            cursor: pointer;
        }
        .nav-item.active {
            color: #2e86c1;
            border-bottom-color: #2e86c1;
        }
        .status-badge {
            background-color: #e8f8f5;
            color: #17a589;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #17a589;
        }
        
        /* Khung chứa kết quả bên phải */
        .result-placeholder {
            background-color: var(--secondary-background-color);
            border: 2px dashed rgba(128,128,128,0.2);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            color: #7f8c8d;
            text-align: center;
        }
        
        /* Nút Bắt đầu chẩn đoán */
        div.stButton > button {
            background-color: #5b2c6f;
            color: white;
            border-radius: 8px;
            width: 100%;
            padding: 10px;
            font-weight: bold;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            background-color: #4a235a;
            transform: translateY(-2px);
        }
        
        /* Thẻ kết quả AI */
        .ai-result-card {
            background-color: var(--background-color);
            border-radius: 12px;
            padding: 20px;
            border-left: 6px solid #28b463;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .ai-disease { border-left-color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 3. KHAI BÁO KIẾN TRÚC MÔ HÌNH (GIỮ NGUYÊN)
# ==========================================
class OptimalAttentiveProbe(nn.Module):
    def __init__(self, in_dim=384, num_classes=11, dropout_p=0.1): 
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return self.classifier(self.dropout(x * attn_weights))

class EUPE_ViT_Optimal(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        self.head = OptimalAttentiveProbe(in_dim=384, num_classes=num_classes)
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_model():
    model = EUPE_ViT_Optimal(num_classes=11)
    model.load_state_dict(torch.load('eupe_vits16_best_stage2.pth', map_location=torch.device('cpu')))
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
        img = img_bgr[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def generate_heatmap(model, tensor_img, original_img):
    img_np = np.array(original_img)
    attentions = model.backbone.get_last_selfattention(tensor_img)
    cls_attn = attentions[0, :, 0, 1:].reshape(attentions.shape[0], tensor_img.shape[-2]//16, tensor_img.shape[-1]//16)
    attn_map = cls_attn.mean(dim=0).detach().cpu().numpy()
    attn_map = cv2.resize((attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8), (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0))

class_names = ['Advanced/End-stage Glaucoma', 'Dry Age-Related Macular Degeneration', 'Mild Diabetic Retinopathy', 'Mild Glaucoma', 'Moderate Diabetic Retinopathy', 'Moderate Glaucoma', 'No Age-Related Macular Degeneration (Khỏe mạnh)', 'Proliferative Diabetic Retinopathy (PDR)', 'Severe Diabetic Retinopathy', 'Severe Glaucoma', 'Wet Age-Related Macular Degeneration']
eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ==========================================
# 4. GIAO DIỆN HIỂN THỊ
# ==========================================

if st.session_state.current_page == 'home':
    # --- GIAO DIỆN TRANG CHỦ ---
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>AI CHẨN ĐOÁN VÕNG MẠC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; opacity: 0.9;'>Hệ thống chẩn đoán 11 bệnh lý đáy mắt sử dụng kiến trúc Học Sâu (Deep Learning). Độ chính xác cao, hỗ trợ bác sĩ ra quyết định lâm sàng nhanh chóng.</p>", unsafe_allow_html=True)
    
    if st.button("🚀 BẮT ĐẦU SỬ DỤNG HỆ THỐNG"):
        change_page('diagnostic')
        st.rerun()

else:
    # --- GIAO DIỆN TRANG CHẨN ĐOÁN (DASHBOARD) ---
    
    # 1. Thanh Top Navigation ảo giống ảnh thiết kế
    st.markdown("""
    <div class='top-nav'>
        <div style='display: flex; gap: 20px; align-items: center;'>
            <div style='font-size: 1.2rem; font-weight: 900; color: #2874a6;'>👁️ EUPE-ViT AI</div>
            <div class='nav-item active'>1. Phân tích ảnh</div>
            <div class='nav-item'>2. Báo cáo thống kê</div>
            <div class='nav-item'>3. Hồ sơ bệnh nhân</div>
        </div>
        <div class='status-badge'>🟢 HỆ THỐNG SẴN SÀNG</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Nút quay lại trang chủ
    if st.button("🔙 Quay lại Trang Chủ"):
        change_page('home')
        st.rerun()
        
    st.markdown("<h2>Chẩn đoán bệnh lý võng mạc</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-top: -10px;'>Tải ảnh nội soi đáy mắt lên để hệ thống phân tích tự động.</p>", unsafe_allow_html=True)
    st.write("")

    # 2. Bố cục chia cột 1:2
    col_left, col_right = st.columns([1.2, 2.5])
    
    with col_left:
        st.markdown("#### 📂 Tệp hình ảnh đầu vào")
        st.info("Hỗ trợ định dạng: PNG, JPG, JPEG (Tối đa 200MB)")
        uploaded_file = st.file_uploader("Tải ảnh", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        st.write("")
        st.markdown("Chọn nút bên dưới để hệ thống nhận diện chính xác bệnh lý.")
        predict_button = st.button("🔍 Tiến hành phân tích", disabled=(uploaded_file is None))

    with col_right:
        if uploaded_file is None:
            # Giao diện chờ giống ảnh 2
            st.markdown("""
            <div class='result-placeholder'>
                <div style='font-size: 50px; background: #eaf2f8; padding: 20px; border-radius: 50%; color: #3498db; margin-bottom: 15px;'>🔍</div>
                <h3>Khu vực kết quả phân tích</h3>
                <p>Sau khi tải ảnh và bắt đầu phân tích, kết quả chẩn đoán và bản đồ chú ý sẽ xuất hiện tại đây.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            original_image = Image.open(uploaded_file).convert('RGB')
            cropped_image = crop_fundus(original_image)
            
            if not predict_button:
                # Hiện ảnh gốc trước khi bấm nút
                st.markdown("#### Hình ảnh đang chờ xử lý")
                st.image(original_image, use_container_width=True)
            else:
                with st.spinner('⏳ AI đang xử lý điểm ảnh...'):
                    try:
                        model = load_model()
                        tensor_img = eval_transform(cropped_image).unsqueeze(0)
                        with torch.no_grad():
                            outputs = model(tensor_img)
                            probs = F.softmax(outputs, dim=1)[0].numpy()
                            heatmap_img = generate_heatmap(model, tensor_img, cropped_image)
                        
                        top3_idx = np.argsort(probs)[-3:][::-1]
                        is_healthy = (top3_idx[0] == 6)
                        success = True
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi: {e}")
                        success = False
                
                if success:
                    st.toast('Phân tích thành công!', icon='✅')
                    
                    # Layout kết quả
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("**1. Ảnh đầu vào (Cắt viền đen)**")
                        st.image(cropped_image, use_container_width=True)
                    with res_col2:
                        st.markdown("**2. Bản đồ chú ý vùng bệnh (Heatmap)**")
                        st.image(heatmap_img, use_container_width=True)
                    
                    st.write("---")
                    st.markdown("#### 📝 Báo cáo Chẩn đoán chi tiết")
                    card_status = "" if is_healthy else "ai-disease"
                    color_hex = "#28b463" if is_healthy else "#e74c3c"
                    
                    st.markdown(f"""
                    <div class='ai-result-card {card_status}'>
                        <div style='color: #7f8c8d; font-weight: bold;'>BỆNH LÝ CHÍNH ĐƯỢC PHÁT HIỆN:</div>
                        <h2 style='color: {color_hex}; margin-top: 5px; margin-bottom: 5px;'>{class_names[top3_idx[0]]}</h2>
                        <div style='display: inline-block; background: #f2f3f4; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>Độ tin cậy: {probs[top3_idx[0]]*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Các nguy cơ liên đới:**")
                    for i in range(1, 3):
                        idx = top3_idx[i]
                        st.caption(f"{class_names[idx]} (**{probs[idx]*100:.2f}%**)")
                        st.progress(int(probs[idx]*100))