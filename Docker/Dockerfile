FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

RUN pip install --upgrade pip

# ComfyUI 설치
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# 기존 최적화된 torch를 유지하기 위해 requirements에서 torch 관련 항목 제외하고 설치
RUN cd /app/ComfyUI && \
    grep -vE "torch|torchvision|torchaudio" requirements.txt > requirements_no_torch.txt && \
    pip install -r requirements_no_torch.txt

# ComfyUI-Manager 설치
RUN mkdir -p /app/ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /app/ComfyUI/custom_nodes/ComfyUI-Manager && \
    cd /app/ComfyUI/custom_nodes/ComfyUI-Manager && \
    grep -vE "torch|torchvision|torchaudio" requirements.txt > requirements_no_torch.txt || touch requirements_no_torch.txt && \
    pip install -r requirements_no_torch.txt

EXPOSE 8188

CMD ["python", "/app/ComfyUI/main.py", "--enable-cors-header", "*", "--listen", "0.0.0.0", "--port", "8188"]
