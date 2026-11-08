# Docker image for the Gradio demo (ONNX INT8 model, CPU only)
#
# Build:  docker build -t chest-xray-demo .
# Run:    docker run -p 7860:7860 chest-xray-demo
# Open:   http://localhost:7860

FROM python:3.12-slim

WORKDIR /app

COPY demo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo/app.py .

# The trained INT8 model is not stored in git (too big), so download it from the Hugging Face Space
ADD https://huggingface.co/spaces/Mhiu/chest-xray-ai/resolve/main/model_int8.onnx model_int8.onnx

# Gradio must listen on all interfaces so the browser outside the container can reach it
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "app.py"]
