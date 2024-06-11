import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st

def preprocess_image(img, target_size=(28, 28)):
    # Chuyển đổi ảnh thành ảnh xám
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tính giá trị trung bình của ảnh xám
    avg_pixel_value = np.mean(gray_img)
    
    # Xác định xem nền của ảnh có gần màu trắng không (nền là màu sáng)
    if avg_pixel_value > 120:  # Điều kiện có thể thay đổi tùy thuộc vào ngưỡng màu sáng bạn muốn
        # Đảo ngược màu ảnh nhị phân
        inverted_img = cv2.bitwise_not(gray_img)
    else:
        inverted_img = gray_img
    
    # Áp dụng Thresholding để chuyển đổi thành ảnh nhị phân
    _, binary_img = cv2.threshold(inverted_img, 127, 255, cv2.THRESH_BINARY)
    
    # Chuẩn hoá kích thước ảnh
    resized_img = cv2.resize(binary_img, target_size)
    
    return resized_img

# Load model
model = load_model('rnn_model.keras')  # Ensure the model file name is correct

# Character lookup dictionary
char_dict = {i: char for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
char_dict[len(char_dict)] = '?'

# Streamlit application
def main():
    st.set_page_config(page_title="Nhận dạng chữ viết tay", page_icon="🔤", layout="wide")
    
    st.markdown("""
        <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-container {
            border: 2px solid #e6e6e6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-container {
            border: 2px solid #e6e6e6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            font-size: 30px;
        }
        .prediction-text {
            font-size: 35px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<div class='title'>Nhận dạng chữ viết tay</div>", unsafe_allow_html=True)
    st.markdown("<div class='upload-container'><div>Tải lên một ảnh để chuyển đổi:</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(uploaded_file, caption='Ảnh đã tải lên', use_column_width=False, width=200)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Đọc ảnh từ file và chuyển thành mảng numpy
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        # Tiền xử lý ảnh
        preprocessed_img = preprocess_image(image)
        
        # Hiển thị ảnh đã tiền xử lý
        st.image(preprocessed_img, caption='Ảnh sau khi tiền xử lý', use_column_width=False, width=200)
        
        input_data = np.expand_dims(preprocessed_img, axis=(0, -1)) / 255.0
        
        with st.spinner('Predicting...'):
            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)

        if predicted_index in char_dict:
            predicted_character = char_dict[predicted_index]
        else:
            predicted_character = '?'

        st.markdown(f"<div class='result-container'>Kí tự : <span class='prediction-text'>{predicted_character}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
