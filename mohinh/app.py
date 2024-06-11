import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Khử nhiễu ảnh bằng Bilateral Filtering
def bilateral_filtering(image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return cv2.bilateralFilter(image, 9, 75, 75)

# Chuyển ảnh thành ảnh xám
def grayscale(image):
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image

# Chuyển ảnh thành ảnh nhị phân
def thresholding(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh

# Kiểm tra nền sáng/tối và đảo ngược màu ảnh nếu cần
def check_background(image):
    bg_value = np.mean(image)
    if bg_value > 127:
        return image
    else:
        return cv2.bitwise_not(image)

import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np

# Hàm cắt chữ từ ảnh và chuẩn hoá kích thước thành 28x28 pixel và căn giữa vào ô 20x20 pixel
def crop_characters(image):
    inverted_image = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_characters = []
    target_size = (28, 28)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        character = inverted_image[y:y+h, x:x+w]
        
        # Chuẩn hoá kích thước thành 28x28 pixel
        resized_char = cv2.resize(character, target_size)
        
        # Tạo ảnh mới 28x28 và căn giữa ký tự vào
        new_img = np.zeros(target_size, dtype=np.uint8)
        start_x = (target_size[1] - resized_char.shape[1]) // 2
        start_y = (target_size[0] - resized_char.shape[0]) // 2
        new_img[start_y:start_y + resized_char.shape[0], start_x:start_x + resized_char.shape[1]] = resized_char
        
        cropped_characters.append((x, new_img))
    
    cropped_characters.sort(key=lambda x: x[0])
    return [char for _, char in cropped_characters]



# Hàm dự đoán kí tự từ ảnh
def predict_character(model, character_image):
    input_data = np.expand_dims(character_image, axis=(0, -1)) / 255.0
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if predicted_index < len(alphabet):
        return alphabet[predicted_index]
    else:
        return '?'

def main():
    st.title("Ứng dụng nhận diện chữ viết tay")

    # Đường dẫn đến tệp tin chứa mô hình
    model_path = "rnn_model.keras"  # Thay đổi đường dẫn tới mô hình của bạn tại đây

    # Đọc mô hình nhận diện chữ viết tay
    model = load_model(model_path)

    uploaded_image = st.file_uploader("Tải lên ảnh", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        # Đọc ảnh từ file đã tải lên
        image = Image.open(uploaded_image)
        st.image(image, caption='Ảnh gốc', use_column_width=True)

        # Chuyển ảnh thành mảng numpy
        image_np = np.array(image)

        # Tiền xử lý ảnh
        processed_image = bilateral_filtering(image_np)
        processed_image = grayscale(processed_image)
        processed_image = thresholding(processed_image)
        processed_image = check_background(processed_image)
        cropped_characters = crop_characters(processed_image)

        # Lấy tọa độ x của hộp giới hạn cho mỗi kí tự
        character_coords = [(cv2.boundingRect(char)[0], char) for char in cropped_characters]

        # Sắp xếp các kí tự từ trái qua phải
        character_coords.sort(key=lambda x: x[0])  # Sắp xếp theo tọa độ x (top-left)

        # Ghép các kí tự thành chữ hoàn chỉnh
        predicted_text = ""
        for _, character_image in character_coords:
            predicted_character = predict_character(model, character_image)
            predicted_text += predicted_character

        # Hiển thị chữ hoàn chỉnh
        st.write("Chữ nhận dạng:", predicted_text)

        # Hiển thị các kí tự đã cắt
        st.markdown("### Các kí tự được cắt:")
        for i, (x, character_image) in enumerate(character_coords, start=1):
            st.image(character_image, caption=f"Kí tự {i}", use_column_width=True)

if __name__ == "__main__":
    main()
