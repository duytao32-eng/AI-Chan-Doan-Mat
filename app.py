import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & QUẢN LÝ TRẠNG THÁI (ROUTING)
# ==========================================
st.set_page_config(page_title="AI Chẩn Đoán Võng Mạc", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def change_page(page_name):
    st.session_state.current_page = page_name

# ==========================================
# 2. CSS DYNAMIC TÙY THEO TRANG
# ==========================================
if st.session_state.current_page == 'home':
    # CSS CHO TRANG CHỦ - ÉP BUỘC HIỂN THỊ ẢNH BÁC SĨ VÀ LÀM MƯỢT NÚT BẤM
    st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-photo/medical-technology-concept-with-doctor-touching-virtual-screen_53876-104054.jpg") !important;
            background-size: cover !important;
            background-attachment: fixed !important;
            background-position: center !important;
        }
        .block-container {
            background-color: rgba(15, 32, 39, 0.85); /* Nền tối mờ sang trọng */
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 20px;
            padding: 4rem 3rem;
            margin-top: 8vh;
            max-width: 900px;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 { color: #ffffff !important; font-size: 3.5rem !important; margin-bottom: 20px !important; letter-spacing: 1px; }
        .intro-text { color: #e0e0e0 !important; font-size: 1.25rem !important; line-height: 1.8 !important; text-align: justify; margin-bottom: 30px; }
        
        /* Hiệu ứng nút bấm mượt mà */
        div.stButton > button {
            background: linear-gradient(135deg, #1f77b4, #2874A6);
            color: white !important;
            border-radius: 30px;
            padding: 15px 40px;
            font-size: 20px !important;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4);
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 30px rgba(31, 119, 180, 0.7);
            background: linear-gradient(135deg, #2874A6, #1f77b4);
        }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

else:
    # CSS CHO TRANG CHẨN ĐOÁN
    st.markdown("""
    <style>
        .stApp { background-image: none !important; }
        .block-container { padding-top: 2rem; max-width: 1400px; }
        
        .top-nav {
            display: flex;
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
            cursor: default;
        }
        .nav-item.active { color: #2e86c1; border-bottom-color: #2e86c1; }
        
        .result-placeholder {
            background-color: var(--secondary-background-color);
            border: 2px dashed rgba(128,128,128,0.3);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 450px;
            color: var(--text-color);
            text-align: center;
            padding: 20px;
        }
        
        div.stButton > button {
            background-color: #34495e;
            color: white;
            border-radius: 8px;
            width: 100%;
            padding: 12px;
            font-weight: bold;
            transition: all 0.3s;
        }
        div.stButton > button:hover { background-color: #2c3e50; transform: translateY(-2px); }
        
        .ai-result-card {
            background-color: var(--secondary-background-color);
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
# 3. KHAI BÁO KIẾN TRÚC MÔ HÌNH
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

# ĐÃ SỬA LỖI SHAPE TENSOR TẠI ĐÂY
def generate_heatmap(model, tensor_img, original_img):
    img_np = np.array(original_img)
    attentions = model.backbone.get_last_selfattention(tensor_img) # shape: (1, 6, 197, 197)
    num_heads = attentions.shape[1]
    
    # Bỏ token CLS đầu tiên, lấy attention của 196 patches còn lại
    cls_attn = attentions[0, :, 0, 1:] # shape: (6, 196)
    
    w_featmap = tensor_img.shape[-1] // 16 # 224//16 = 14
    h_featmap = tensor_img.shape[-2] // 16 # 224//16 = 14
    
    # Reshape đúng kích thước: (số head, chiều cao, chiều rộng)
    cls_attn = cls_attn.reshape(num_heads, h_featmap, w_featmap) 
    attn_map = cls_attn.mean(dim=0).detach().cpu().numpy() # Lấy trung bình các head -> (14, 14)
    
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = cv2.resize(attn_map, (img_np.shape[1], img_np.shape[0]))
    
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0))

class_names = ['Advanced/End-stage Glaucoma', 'Dry Age-Related Macular Degeneration', 'Mild Diabetic Retinopathy', 'Mild Glaucoma', 'Moderate Diabetic Retinopathy', 'Moderate Glaucoma', 'No Age-Related Macular Degeneration (Khỏe mạnh)', 'Proliferative Diabetic Retinopathy (PDR)', 'Severe Diabetic Retinopathy', 'Severe Glaucoma', 'Wet Age-Related Macular Degeneration']
eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ==========================================
# 4. GIAO DIỆN HIỂN THỊ
# ==========================================

if st.session_state.current_page == 'home':
    # --- GIAO DIỆN TRANG CHỦ (ĐÃ LÀM DÀI VÀ CHUYÊN NGHIỆP HƠN) ---
    st.markdown("<h1>HỆ THỐNG CHẨN ĐOÁN VÕNG MẠC AI</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intro-text'>
        Đồ án nghiên cứu ứng dụng mô hình <b>Học sâu (Deep Learning)</b> với kiến trúc mạng <b>Gated EUPE (Vision Transformer)</b> tiên tiến. 
        Hệ thống được huấn luyện trên tập dữ liệu y khoa chuẩn xác, có khả năng nhận diện tự động và phân loại <b>11 bệnh lý đáy mắt phức tạp</b> (như Võng mạc tiểu đường, Tăng nhãn áp, Thoái hóa điểm vàng...). <br><br>
        Đây là giải pháp công nghệ hỗ trợ đắc lực cho các y bác sĩ trong quá trình tầm soát, tối ưu hóa quy trình khám chữa bệnh và ra quyết định lâm sàng nhanh chóng, chính xác.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🚀 BẮT ĐẦU PHÂN TÍCH NGAY"):
            change_page('diagnostic')
            st.rerun()

else:
    # --- GIAO DIỆN TRANG CHẨN ĐOÁN ---
    # Đã xóa dòng "Hệ thống sẵn sàng" bên góc phải
    st.markdown("""
    <div class='top-nav'>
        <div style='display: flex; gap: 20px; align-items: center;'>
            <div style='font-size: 1.3rem; font-weight: 900; color: #2874a6;'>👁️ EUPE-ViT AI</div>
            <div class='nav-item active'>1. Phân tích ảnh nội soi</div>
            <div class='nav-item'>2. Báo cáo thống kê</div>
            <div class='nav-item'>3. Hồ sơ y tế</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        if st.button("🔙 Quay lại"):
            change_page('home')
            st.rerun()
        
    st.markdown("<h2>Giao diện phân tích bệnh lý</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7f8c8d; margin-top: -10px;'>Cung cấp hình ảnh soi đáy mắt để thuật toán trích xuất đặc trưng và đánh giá rủi ro.</p>", unsafe_allow_html=True)
    st.write("")

    col_left, col_right = st.columns([1.2, 2.5])
    
    with col_left:
        st.markdown("#### 📂 Dữ liệu đầu vào")
        st.info("Định dạng cho phép: PNG, JPG, JPEG")
        uploaded_file = st.file_uploader("Tải ảnh", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        st.write("")
        st.markdown("Click vào nút bên dưới để khởi chạy quy trình phân tích của AI.")
        predict_button = st.button("🔍 Khởi chạy Trí tuệ nhân tạo", disabled=(uploaded_file is None))

    with col_right:
        if uploaded_file is None:
            # Viết lại hoàn toàn nội dung để không giống bản gốc
            st.markdown("""
            <div class='result-placeholder'>
                <div style='font-size: 55px; background: rgba(52, 152, 219, 0.1); padding: 20px; border-radius: 50%; margin-bottom: 20px;'>🔬</div>
                <h3>Bảng Điều Khiển AI</h3>
                <p style='font-size: 1.1rem; opacity: 0.8; max-width: 80%;'>Hệ thống đang chờ dữ liệu. Vui lòng tải lên ảnh chụp võng mạc từ menu bên trái để thuật toán tiến hành chẩn đoán và xuất bản đồ tổn thương.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            original_image = Image.open(uploaded_file).convert('RGB')
            cropped_image = crop_fundus(original_image)
            
            if not predict_button:
                st.markdown("#### Hình ảnh đang chờ xử lý")
                st.image(original_image, use_container_width=True)
            else:
                with st.spinner('⏳ Thuật toán đang quét và đối chiếu các điểm ảnh bất thường...'):
                    try:
                        model = load_model()
                        tensor_img = eval_transform(cropped_image).unsqueeze(0)
                        with torch.no_grad():
                            outputs = model(tensor_img)
                            probs = F.softmax(outputs, dim=1)[0].numpy()
                            # Heatmap nay đã hoạt động bình thường
                            heatmap_img = generate_heatmap(model, tensor_img, cropped_image)
                        
                        top3_idx = np.argsort(probs)[-3:][::-1]
                        is_healthy = (top3_idx[0] == 6)
                        success = True
                    except Exception as e:
                        st.error(f"Lỗi hệ thống: {e}")
                        success = False
                
                if success:
                    st.toast('Phân tích thành công!', icon='✅')
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("**Ảnh đã qua tiền xử lý (Cắt viền)**")
                        st.image(cropped_image, use_container_width=True)
                    with res_col2:
                        st.markdown("**Bản đồ vùng chú ý của AI (Heatmap)**")
                        st.image(heatmap_img, use_container_width=True)
                    
                    st.write("---")
                    st.markdown("#### 📝 Kết luận lâm sàng từ AI")
                    card_status = "" if is_healthy else "ai-disease"
                    color_hex = "#28b463" if is_healthy else "#e74c3c"
                    
                    st.markdown(f"""
                    <div class='ai-result-card {card_status}'>
                        <div style='color: #7f8c8d; font-weight: bold;'>NGUY CƠ CAO NHẤT:</div>
                        <h2 style='color: {color_hex}; margin-top: 5px; margin-bottom: 5px;'>{class_names[top3_idx[0]]}</h2>
                        <div style='display: inline-block; background: rgba(128,128,128,0.1); padding: 5px 15px; border-radius: 20px; font-weight: bold;'>Độ tin cậy: {probs[top3_idx[0]]*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Các nguy cơ tiềm ẩn khác:**")
                    for i in range(1, 3):
                        idx = top3_idx[i]
                        st.caption(f"{class_names[idx]} (**{probs[idx]*100:.2f}%**)")
                        st.progress(int(probs[idx]*100))