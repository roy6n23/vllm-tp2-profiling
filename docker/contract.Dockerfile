FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 VLLM_TARGET_DEVICE=cpu
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install vllm==0.30.0 --no-deps
COPY docker/common.txt /tmp/common.txt
RUN pip install -r /tmp/common.txt pytest numpy psutil
RUN python -c "from huggingface_hub import snapshot_download as s; s('NousResearch/Meta-Llama-3.1-8B-Instruct', revision='d10aef7999a2b5ba950ab3974312feeedbfe0b77', allow_patterns=['tokenizer*','special_tokens_map.json','config.json','generation_config.json'], local_dir='/opt/tok')"
COPY . /repo
WORKDIR /repo
CMD ["python", "-m", "pytest", "-q", "-m", "contract", "tests/contract"]
