{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6e895e86-a456-47e0-948f-ac9c1707fd16",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "(unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \\UXXXXXXXX escape (4158666715.py, line 28)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[5], line 28\u001b[1;36m\u001b[0m\n\u001b[1;33m    model = load_model('C:\\Users\\tramy\\Downloads\\AI_btl\\mohinh.keras')  # Thay đổi đường dẫn đến mô hình của bạn\u001b[0m\n\u001b[1;37m                                                                     ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \\UXXXXXXXX escape\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "def adjust_brightness_and_resize(img, target_size=(28, 28)):\n",
    "    # Chuyển đổi ảnh từ Streamlit uploader thành một mảng numpy\n",
    "    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)\n",
    "    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)\n",
    "    \n",
    "    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    # Tính giá trị trung bình của các pixel\n",
    "    avg_pixel_value = np.mean(gray_img)\n",
    "    \n",
    "    # Xác định ngưỡng dựa trên giá trị trung bình và phân loại ảnh\n",
    "    if avg_pixel_value > 127:  # Nếu giá trị trung bình lớn hơn ngưỡng 127, ảnh có nền sáng\n",
    "        inverted_img = cv2.bitwise_not(gray_img)  # Đảo ngược màu ảnh\n",
    "    else:  # Ngược lại, ảnh có nền tối\n",
    "        inverted_img = gray_img\n",
    "    \n",
    "    # Resize ảnh\n",
    "    resized_img = cv2.resize(inverted_img, target_size)\n",
    "    resized_img = np.expand_dims(resized_img, axis=-1)\n",
    "    return resized_img\n",
    "\n",
    "# Load the Keras model for character prediction\n",
    "model = load_model('C:\\Users\\tramy\\Downloads\\AI_btl\\mohinh.keras')  # Thay đổi đường dẫn đến mô hình của bạn\n",
    "\n",
    "# Define the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Character Recognition App\")\n",
    "    st.write(\"Upload an image to recognize a character.\")\n",
    "\n",
    "    # File uploader to upload the image\n",
    "    uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"png\", \"jpeg\"])\n",
    "\n",
    "    if uploaded_file is not None:\n",
    "        # Adjust brightness and resize the uploaded image\n",
    "        adjusted_img = adjust_brightness_and_resize(uploaded_file)\n",
    "\n",
    "        # Prepare input data for the model\n",
    "        input_data = np.expand_dims(adjusted_img, axis=0) / 255.0\n",
    "\n",
    "        # Predict\n",
    "        prediction = model.predict(input_data)\n",
    "        predicted_index = np.argmax(prediction)\n",
    "\n",
    "        # Map the predicted index to the corresponding character\n",
    "        char_dict = {i: char for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}\n",
    "        char_dict[len(char_dict)] = '?'  # Special character if index not found\n",
    "        predicted_character = char_dict.get(predicted_index, '?')\n",
    "\n",
    "        # Show the predicted character\n",
    "        st.write(f'Predicted character: {predicted_character}')\n",
    "\n",
    "        # Display the adjusted image\n",
    "       st.image(adjusted_img.squeeze(), caption='Adjusted Image', use_column_width=True)\n",
    "\n",
    "# Run the Streamlit app\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e87c98af-5d3c-4656-b509-8053b3f50c2b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
