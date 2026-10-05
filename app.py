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
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card {
        background-color: var(--secondary-background-color);
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid;
        animation: fadeSlideUp 0.5s ease-out forwards;
        margin: 10px 0px 25px 0px;
    }
    .healthy { border-left-color: #00cc66; } 
    .disease { border-left-color: #ff4b4b; } 
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
    img {
        border-radius: 8px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    img:hover { transform: scale(1.02); }
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
# 3. HÀM XỬ LÝ ẢNH & ĐỌC MÔ HÌNH
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

def generate_heatmap(model, tensor_img, original_img):
    """Trích xuất Attention của ViT để vẽ Heatmap"""
    img_np = np.array(original_img)
    # Lấy attention weights từ block cuối cùng của mạng DINO ViT
    attentions = model.backbone.get_last_selfattention(tensor_img)
    # Tách attention của CLS token (index 0) so với các patch (từ index 1)
    cls_attn = attentions[0, :, 0, 1:] 
    
    w_featmap = tensor_img.shape[-1] // 16 # 224/16 = 14
    h_featmap = tensor_img.shape[-2] // 16 # 224/16 = 14
    
    cls_attn = cls_attn.reshape(cls_attn.shape[0], h_featmap, w_featmap)
    # Lấy trung bình tất cả các head
    attn_map = cls_attn.mean(dim=0).detach().cpu().numpy()
    
    # Normalize bản đồ nhiệt
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = cv2.resize(attn_map, (img_np.shape[1], img_np.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Chồng Heatmap lên ảnh gốc với độ mờ 50%
    result = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return Image.fromarray(result)

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
# 4. GIAO DIỆN CHÍNH
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865744.png", width=80)
    st.markdown("## THÔNG TIN ĐỒ ÁN")
    st.info("Hệ thống dự đoán đa lớp bệnh lý võng mạc thông qua ảnh nội soi đáy mắt.")
    st.markdown("👨‍‍🎓 **SVTH:** [Tên của bạn]")
    st.markdown("👨‍🏫 **GVHD:** [Tên Giáo viên]")

st.markdown("<h1 style='text-align: center;'>👁️ AI CHẨN ĐOÁN VÕNG MẠC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; opacity: 0.7;'>Phát hiện tự động 11 loại bệnh lý về mắt với độ chính xác cao.</p>", unsafe_allow_html=True)
st.write("")

# Khu vực Upload
col_up1, col_up2, col_up3 = st.columns([1, 4, 1])
with col_up2:
    uploaded_file = st.file_uploader("📥 Tải ảnh đáy mắt lên tại đây (Hỗ trợ: PNG, JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    st.divider()
    original_image = Image.open(uploaded_file).convert('RGB')
    cropped_image = crop_fundus(original_image)
    
    col1, col2, col3 = st.columns([1.2, 0.2, 1.5])
    
    # ---------------- BÊN TRÁI: HÌNH ẢNH ----------------
    with col1:
        st.subheader("🖼️ Quá trình xử lý ảnh", anchor=False)
        st.write("**📷 Ảnh Gốc**")
        st.image(original_image, use_container_width=True)
        st.write("**⚙️ Đã Cắt Viền Đen (Crop)**")
        st.image(cropped_image, use_container_width=True)
        
    # ---------------- BÊN PHẢI: PHÂN TÍCH & KẾT QUẢ ----------------
    with col3:
        st.subheader("📊 Kết Quả Chẩn Đoán AI", anchor=False)
        predict_button = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner('⏳ Hệ thống AI đang phân tích dữ liệu...'):
                try:
                    model = load_model()
                    tensor_img = eval_transform(cropped_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = model(tensor_img)
                        probs = F.softmax(outputs, dim=1)[0].numpy()
                        # Tạo Heatmap bằng Attention của mô hình DINO ViT
                        heatmap_img = generate_heatmap(model, tensor_img, cropped_image)
                    
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    is_healthy = (top3_idx[0] == 6)
                    success = True
                    
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
                    success = False
            
            if success:
                st.toast('Hoàn tất phân tích!', icon='✅')
                if is_healthy: st.balloons()
                
                # CHIA TABS KẾT QUẢ VÀ HEATMAP
                tab1, tab2 = st.tabs(["📝 Chẩn Đoán & Báo Cáo", "🔥 Bản Đồ Chú Ý (AI Heatmap)"])
                
                with tab1:
                    card_status = "healthy" if is_healthy else "disease"
                    top1_name = class_names[top3_idx[0]]
                    top1_conf = probs[top3_idx[0]] * 100
                    color_hex = "#00cc66" if is_healthy else "#ff4b4b"
                    
                    st.markdown(f"""
                    <div class='result-card {card_status}'>
                        <div style='font-size: 1.1rem; opacity: 0.7; margin-bottom: 5px; text-transform: uppercase;'>Nguy cơ cao nhất:</div>
                        <div style='font-size: 1.8rem; font-weight: 800; color: {color_hex}; line-height: 1.2;'>{top1_name}</div>
                        <div class='confidence-badge'>Độ tin cậy: {top1_conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Các nguy cơ khác:**")
                    for i in range(1, 3):
                        idx = top3_idx[i]
                        st.caption(f"{class_names[idx]} (**{probs[idx]*100:.2f}%**)")
                        st.progress(int(probs[idx]*100))
                    
                    # --- NÚT XUẤT BÁO CÁO Y TẾ (.TXT) ---
                    st.write("")
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    report_content = f"""
===================================================
      BÁO CÁO CHẨN ĐOÁN VÕNG MẠC BẰNG TRÍ TUỆ NHÂN TẠO
===================================================
Thời gian phân tích: {current_time}
Kiến trúc AI sử dụng: Gated EUPE (DINO ViT-S/16)
                    
KẾT QUẢ DỰ ĐOÁN CHÍNH:
>> {top1_name} (Độ tin cậy: {top1_conf:.2f}%)
                    
CÁC NGUY CƠ TIỀM ẨN:
- {class_names[top3_idx[1]]} ({probs[top3_idx[1]]*100:.2f}%)
- {class_names[top3_idx[2]]} ({probs[top3_idx[2]]*100:.2f}%)
                    
* LƯU Ý: Đây là hệ thống AI hỗ trợ chẩn đoán chuyên môn. 
Vui lòng kết hợp thăm khám lâm sàng cùng bác sĩ để có kết luận chính xác nhất.
===================================================
                    """
                    st.download_button(
                        label="📥 TẢI XUỐNG PHIẾU KẾT QUẢ (TXT)",
                        data=report_content,
                        file_name=f"Bao_Cao_AI_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                with tab2:
                    st.info("Vùng màu **đỏ/cam** là các khu vực tổn thương (hoặc bất thường) mà Trí tuệ nhân tạo (AI) đang tập trung cao nhất để đưa ra kết luận.")
                    st.image(heatmap_img, caption="Bản đồ chú ý (Self-Attention Heatmap)", use_container_width=True)