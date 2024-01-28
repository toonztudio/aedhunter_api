# FROM python:3.10

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev

# WORKDIR /src

# ADD ./ /src

# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8000

# CMD ["python", "main.py"]

FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

WORKDIR /src

ADD ./ /src

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy SSL certificates to the container
COPY ./ssl/privkey.pem /etc/ssl/private/
COPY ./ssl/fullchain.pem /etc/ssl/certs/

EXPOSE 8000

CMD ["python", "main.py", "--certfile", "/etc/ssl/certs/fullchain.pem", "--keyfile", "/etc/ssl/private/privkey.pem", "--bind", "0.0.0.0:8000", "--workers", "4", "--access-logfile", "-", "--error-logfile", "-"]