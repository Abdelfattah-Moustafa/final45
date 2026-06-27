FROM python:3.12-slim

# System dependencies for OpenCV / MediaPipe.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LiteRT's bundled .so marks its stack as executable, which Render's kernel
# blocks (seccomp NX policy). Clear PT_GNU_STACK exec flag so the dynamic
# linker can load the library without needing exec-stack permission.
RUN find /usr/local/lib -name "_pywrap_tensorflow_interpreter_wrapper.so" \
    -exec patchelf --clear-execstack {} \; 2>/dev/null || true

COPY . .

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

# Apply migrations + load data, then start the server (see scripts/entrypoint.sh).
CMD ["./scripts/entrypoint.sh"]
