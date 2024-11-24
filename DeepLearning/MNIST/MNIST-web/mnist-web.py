from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# 加载预训练的模型
model = tf.keras.models.load_model('mnist_model.h5')
model.summary()  # 打印模型摘要

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端发送的图像数据
    data = request.json['image']
    image_data = data.split(',')[1]
    
    # 解码Base64图像数据
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # 将图像转换为灰度图并调整大小为28x28
    image = image.convert('L').resize((28, 28))
    
    # 将图像数据转换为numpy数组并归一化
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    
    # 打印图像数组和形状
    print("Image Array:")
    print(image_array)
    print("Image Array Shape:")
    print(image_array.shape)
    
    # 进行预测
    prediction = model.predict(image_array)
    
    # 打印预测结果
    print("Prediction:")
    print(prediction)
    
    predicted_class = np.argmax(prediction)
    
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)