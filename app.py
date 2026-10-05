import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & QUẢN LÝ TRẠNG THÁI
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
    st.markdown("""
        <style>
            .stApp { background-color: transparent !important; }
            [data-testid="stHeader"] { display: none !important; }
            
            #background-image {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-image: url("https://t3.ftcdn.net/jpg/00/79/70/58/360_F_79705868_f21hI0uSihy1yJq0H7sFmB2tO027q8bU.jpg");
                background-size: cover;
                background-position: center;
                z-index: -1; 
            }
            
            /* Đã đưa nút bấm vào chung khối này để không bị văng */
            .glass-box {
                background-color: rgba(15, 32, 39, 0.85); 
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 4rem 3rem;
                margin: 15vh auto;
                max-width: 900px;
                text-align: center;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.1);
                position: relative;
                z-index: 10;
            }
            
            h1 { color: #ffffff !important; font-size: 3.5rem !important; margin-bottom: 20px !important; letter-spacing: 1px; }
            .intro-text { color: #e0e0e0 !important; font-size: 1.25rem !important; line-height: 1.8 !important; text-align: justify; margin-bottom: 40px; }
            
            /* CSS chung cho nút bấm */
            div.stButton {
                display: flex;
                justify-content: center;
                width: 100%;
            }
            div.stButton > button {
                background: linear-gradient(135deg, #1f77b4, #2874A6) !important;
                color: white !important;
                border-radius: 30px !important;
                padding: 15px 40px !important;
                font-size: 20px !important;
                font-weight: bold !important;
                border: none !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4) !important;
            }
            div.stButton > button:hover {
                transform: scale(1.05) !important;
                box-shadow: 0 12px 30px rgba(31, 119, 180, 0.7) !important;
            }
            [data-testid="collapsedControl"] { display: none; }
        </style>
        <div id="background-image"></div>
    """, unsafe_allow_html=True)

else:
    # --- ĐÃ SỬA LỖI MẤT CHỮ: Sử dụng class mặc định của markdown thay vì ép HTML phức tạp ---
    st.markdown("""
    <style>
        .stApp { background-color: var(--background-color); }
        .block-container { padding-top: 1rem !important; max-width: 1400px; }
        
        /* Thay vì dùng CSS HTML, ta sẽ dùng st.markdown chuẩn của Python ở phần giao diện để hệ thống tự bắt màu */
        
        .result-placeholder {
            background-color: var(--secondary-background-color);
            border: 2px dashed rgba(128,128,128,0.3);
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 420px; 
            text-align: center;
            padding: 20px;
        }
        
        /* Ghi đè lại nút bấm cho trang chẩn đoán để không bị dính hiệu ứng gradient của trang chủ */
        div.stButton > button {
            background: #34495e !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 16px !important;
            width: 100% !important;
            box-shadow: none !important;
        }
        div.stButton > button:hover { 
            background: #2c3e50 !important; 
            transform: translateY(-2px) !important; 
        }
        
        /* Đặc biệt: Nút quay lại cần nhỏ gọn */
        .back-btn-container div.stButton > button {
            width: auto !important;
            padding: 8px 15px !important;
        }
        
        .ai-result-card {
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 20px;
            border-left: 6px solid #28b463;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .ai-disease { border-left-color: #e74c3c; }
        
        .title-compact { font-size: 2rem !important; margin-bottom: 0px !important; padding-bottom: 0px !important;}
        .subtitle-compact { font-size: 1rem !important; margin-top: 5px !important; margin-bottom: 15px !important;}
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
    num_heads = attentions.shape[1]
    cls_attn = attentions[0, :, 0, 1:] 
    
    w_featmap = tensor_img.shape[-1] // 16 
    h_featmap = tensor_img.shape[-2] // 16 
    
    cls_attn = cls_attn.reshape(num_heads, h_featmap, w_featmap) 
    attn_map = cls_attn.mean(dim=0).detach().cpu().numpy()
    
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
    # Nút bấm đã được đưa thẳng vào trong container, sử dụng cột ảo để ép ra giữa
    with st.container():
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        
        st.markdown("<h1>HỆ THỐNG CHẨN ĐOÁN VÕNG MẠC AI</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class='intro-text'>
            Đồ án nghiên cứu ứng dụng mô hình <b>Học sâu (Deep Learning)</b> với kiến trúc mạng <b>Gated EUPE (Vision Transformer)</b> tiên tiến. 
            Hệ thống được huấn luyện trên tập dữ liệu y khoa chuẩn xác, có khả năng nhận diện tự động và phân loại <b>11 bệnh lý đáy mắt phức tạp</b> (như Võng mạc tiểu đường, Tăng nhãn áp, Thoái hóa điểm vàng...). <br><br>
            Đây là giải pháp công nghệ hỗ trợ đắc lực cho các y bác sĩ trong quá trình tầm soát, tối ưu hóa quy trình khám chữa bệnh và ra quyết định lâm sàng nhanh chóng, chính xác.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 BẮT ĐẦU PHÂN TÍCH NGAY"):
            change_page('diagnostic')
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- ĐÃ SỬA: Dùng lệnh st.write kết hợp Markdown chuẩn để không bao giờ bị mất chữ ---
    col_logo, col_nav1, col_nav2, col_nav3, _ = st.columns([2, 2, 2, 2, 4])
    with col_logo:
        st.markdown("**👁️ EUPE-ViT AI**")
    with col_nav1:
        st.markdown("<span style='color: #2e86c1; border-bottom: 2px solid #2e86c1;'>1. Phân tích ảnh nội soi</span>", unsafe_allow_html=True)
    with col_nav2:
        st.write("2. Báo cáo thống kê")
    with col_nav3:
        st.write("3. Hồ sơ y tế")
        
    st.divider() # Tạo đường kẻ phân cách
    
    st.markdown("<div class='back-btn-container'>", unsafe_allow_html=True)
    if st.button("🔙 Quay lại"):
        change_page('home')
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<h2 class='title-compact'>Giao diện phân tích bệnh lý</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-compact'>Cung cấp hình ảnh soi đáy mắt để thuật toán trích xuất đặc trưng và đánh giá rủi ro.</p>", unsafe_allow_html=True)

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
            st.markdown("""
            <div class='result-placeholder'>
                <div style='font-size: 55px; background: rgba(52, 152, 219, 0.1); padding: 20px; border-radius: 50%; margin-bottom: 20px;'>🔬</div>
                <h3>Bảng Điều Khiển AI</h3>
                <p style='font-size: 1.1rem; max-width: 80%;'>Hệ thống đang chờ dữ liệu. Vui lòng tải lên ảnh chụp võng mạc từ menu bên trái để thuật toán tiến hành chẩn đoán và xuất bản đồ tổn thương.</p>
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
                        <div style='font-weight: bold;'>NGUY CƠ CAO NHẤT:</div>
                        <h2 style='color: {color_hex}; margin-top: 5px; margin-bottom: 5px;'>{class_names[top3_idx[0]]}</h2>
                        <div style='display: inline-block; background: rgba(128,128,128,0.1); padding: 5px 15px; border-radius: 20px; font-weight: bold;'>Độ tin cậy: {probs[top3_idx[0]]*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Các nguy cơ tiềm ẩn khác:**")
                    for i in range(1, 3):
                        idx = top3_idx[i]
                        st.caption(f"{class_names[idx]} (**{probs[idx]*100:.2f}%**)")
                        st.progress(int(probs[idx]*100))