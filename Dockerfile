# Base Lambda Python 3.11 image
FROM public.ecr.aws/lambda/python:3.11

# --- OS package updates ---
RUN yum update -y && yum clean all

# --- Install build tools and audio/video dependencies ---
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    sox \
    sox-devel \
    ffmpeg \
    && yum clean all

# --- Copy only requirements first to leverage caching ---
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install -r requirements.txt

# --- Copy BirdNET-Analyzer source code (excluding requirements.txt again) ---
COPY . ${LAMBDA_TASK_ROOT}

# --- Copy Lambda handler separately (to optimize changes) ---
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

# --- Define the Lambda handler command ---
CMD ["lambda_handler.lambda_handler"]
