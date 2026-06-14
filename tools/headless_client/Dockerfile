# Headless Fabric 1.19.4 Minecraft data-collection client.
#
# Renders on the GPU via Xvfb + VirtualGL (EGL backend) so ReplayMod gets a real
# hardware GL context. The blockscope mod auto-connects to the VPS, records a
# session (.mcpr + per-tick state + video), and uploads to Hopper.
#
# Modeled on solaris-engine/camera, but our client is Fabric+mods (not vanilla)
# and ReplayMod produces the video internally, so we drop the VNC/ffmpeg-screen-
# grab stack that Solaris uses.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HOME=/root \
    DISPLAY=:99 \
    VGL_DISPLAY=egl \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xvfb ffmpeg curl wget gnupg ca-certificates \
        fonts-dejavu-core \
        libxi6 libxrender1 libxtst6 libxrandr2 libxext6 libxcursor1 libxxf86vm1 \
        libxv1 libglu1-mesa libegl1 libgl1 libgl1-mesa-dri libglx-mesa0 \
        mesa-utils \
        python3 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# VirtualGL 3.1.1 for GPU-accelerated GL in the headless framebuffer (EGL mode).
RUN curl -fsSL -o /tmp/virtualgl.deb \
        https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb && \
    (dpkg -i /tmp/virtualgl.deb || apt-get update && apt-get install -f -y) && \
    rm -f /tmp/virtualgl.deb

# Java 17 (Temurin) — required by Minecraft 1.19.4.
RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | \
        gpg --dearmor -o /etc/apt/trusted.gpg.d/adoptium.gpg && \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/adoptium.gpg] https://packages.adoptium.net/artifactory/deb jammy main" \
        > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && apt-get install -y --no-install-recommends temurin-17-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir minecraft-launcher-lib==6.5 requests==2.31.0

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
COPY launch_minecraft.py /app/launch_minecraft.py
RUN chmod +x /app/entrypoint.sh

# Mods + config are bind-mounted at runtime from ~/blockscope_client so the jar
# can be re-staged without rebuilding the image:
#   -v ~/blockscope_client/mods:/assets/mods:ro
#   -v ~/blockscope_client/blockscope.properties:/assets/blockscope.properties:ro
ENTRYPOINT ["/app/entrypoint.sh"]
