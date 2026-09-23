# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.14 AS uv
FROM python:3.12.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git unar libarchive-tools xz-utils latexmk texlive-latex-base \
       texlive-latex-extra texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://www.7-zip.org/a/7z2602-linux-x64.tar.xz -o /tmp/7zip.tar.xz \
    && tar -xJf /tmp/7zip.tar.xz -C /usr/local/bin 7zz \
    && rm /tmp/7zip.tar.xz \
    && 7zz i >/dev/null

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked

CMD ["uv", "run", "python", "-m", "mfcontrol", "--help"]
