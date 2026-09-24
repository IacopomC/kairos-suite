# The Kairos evaluation suite.
#
# Adds the suite's own Python dependencies and its code on top of the Kairos
# image, so every script here runs without a separate runtime install.
#
# Build the Kairos image first (see the kairos repo), then from this
# repository's root:
#
#   docker build -t kairos-suite -f docker/Dockerfile .
#
# Run, with your datasets and an output directory mounted:
#
#   docker run --rm -v ~/datasets:/data -v ~/out:/out kairos-suite \
#       python3 evaluation/evaluate.py --method kairos --dataset atc ...
ARG KAIROS_IMAGE=kairos:latest
FROM ${KAIROS_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# The suite plots and tabulates; the pipeline image carries neither. numpy is
# already pinned in the base image and is deliberately not upgraded here, so
# every dependency of matplotlib and pandas is listed explicitly (--no-deps).
RUN pip3 install --no-cache-dir --no-deps \
        "matplotlib==3.7.5" \
        "pandas==2.0.3" \
        "pyyaml" \
        "msgpack" \
        "python-dateutil" "pytz" "tzdata" "cycler" "kiwisolver" "fonttools" "packaging" "pillow" \
        "pyparsing" "contourpy==1.1.1" "importlib-resources" "zipp"

# This checkout replaces the copy of the suite the Kairos image fetched, at the
# path the suite's scripts expect.
RUN rm -rf /ws/src/kairos-suite
COPY . /ws/src/kairos-suite

# The sub-packages import each other by directory name rather than as an
# installed package, so they go on the path directly.
ENV PYTHONPATH=/ws/src/kairos-suite:/ws/src/kairos-suite/evaluation:/ws/src/kairos-suite/grounded_sim:${PYTHONPATH}

WORKDIR /ws/src/kairos-suite
