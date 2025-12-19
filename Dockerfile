# ===== BASE IMAGE (BẮT BUỘC CUDA 12.2) =====
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# ===== SYSTEM DEPENDENCIES =====
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ===== WORKDIR =====
WORKDIR /code

# ===== COPY SOURCE =====
COPY . /code

# ===== INSTALL PYTHON LIBS =====
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# ===== RUN PIPELINE =====
CMD ["bash", "inference.sh"]
