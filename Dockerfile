FROM tensorflow/tensorflow:latest-gpu
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y ffmpeg && pip3 install keras \
								nltk \
								tqdm \
								matplotlib \
								sklearn \
								pillow

