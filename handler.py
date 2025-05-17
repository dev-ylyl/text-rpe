import runpod
import torch
import logging
import time
from transformers import AutoTokenizer, AutoModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载 tokenizer 和 text_model（仅本地加载）
tokenizer = AutoTokenizer.from_pretrained(
    "/runpod-volume/hub/models--BAAI--bge-large-zh-v1.5",
    trust_remote_code=True,
    local_files_only=True
)

text_model = AutoModel.from_pretrained(
    "/runpod-volume/hub/models--BAAI--bge-large-zh-v1.5",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16
).cuda().eval()

# 显式清空初始缓存
torch.cuda.empty_cache()

# 打印当前GPU信息
logging.info(f"🚀 当前使用GPU: {torch.cuda.get_device_name(0)}")

# CUDA预热
with torch.no_grad(), torch.cuda.amp.autocast():
    dummy_inputs = tokenizer(["warmup"], padding=True, return_tensors="pt", truncation=True)
    dummy_inputs = {k: v.cuda() for k, v in dummy_inputs.items()}
    _ = text_model(**dummy_inputs).last_hidden_state.mean(dim=1)
logging.info("✅ 文本模型 warmup 完成")

# 核心处理函数
def handler(job):
    logging.info(f"📥 任务输入内容:\n{job}\n📄 类型: {type(job)}")
    try:
        # 显式清空CUDA缓存
        torch.cuda.empty_cache()
        
        inputs = job["input"].get("data")
        logging.info(f"📋 inputs内容是: {inputs} (类型: {type(inputs)}, 长度: {len(inputs) if inputs else 0})")
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            logging.warning("⚠️ 数据为空")
            return {
                "output": {
                    "error": "Empty input provided."
                }
            }

        start_time = time.time()

        # Tokenizer阶段
        encoded = tokenizer(inputs, padding=True, return_tensors="pt", truncation=True)
        encoded = {k: v.cuda() for k, v in encoded.items()}

        # 推理阶段
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = text_model(**encoded).last_hidden_state.mean(dim=1).cpu().tolist()

        total_time = time.time()
        logging.info(f"✅ 文本推理完成，生成{len(output)}个embedding, 每个维度: {len(output[0])}，总耗时: {total_time - start_time:.3f}s")

        return {
            "output": {
                "embeddings": output
            }
        }

    except Exception as e:
        logging.error(f"❌ 出现异常: {str(e)}")
        torch.cuda.empty_cache()
        return {
            "output": {
                "error": str(e)
            }
        }

# 启动 Serverless Worker
logging.info("🟢 Text Worker 已启动，等待任务中...")
runpod.serverless.start({"handler": handler})