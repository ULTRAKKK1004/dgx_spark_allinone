FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

RUN pip install -r /app/ComfyUI/requirements.txt

# ComfyUI-Manager 설치 추가
RUN mkdir -p /app/ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /app/ComfyUI/custom_nodes/ComfyUI-Manager && \
    pip install -r /app/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

EXPOSE 8188

CMD ["python", "/app/ComfyUI/main.py", "--enable-cors-header", "'*'", "--listen", "0.0.0.0", "--port", "8188"]

