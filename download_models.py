# download_model.py
from huggingface_hub import hf_hub_download

# hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
#                 filename="llama-2-7b-chat.Q4_K_M.gguf",
#                 local_dir="models",
#                 local_dir_use_symlinks=False)

# Option 1: QuantFactory (most popular and actively maintained)
hf_hub_download(repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                local_dir="models")

# Option 2: Alternative repository (uncomment to use instead)
# hf_hub_download(repo_id="MoMonir/Meta-Llama-3-8B-Instruct-GGUF",
#                 filename="meta-llama-3-8b-instruct.Q6_K.gguf",
#                 local_dir="models")

