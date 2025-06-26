# 模型下载
import json
import os
import shutil

from modelscope import snapshot_download

mineru_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
model_dir = snapshot_download("wlc952/mineru", local_dir=f"{mineru_path}/models/mineru")

# 更新 magic-pdf.json 配置文件
config_file = os.path.join(mineru_path, "magic-pdf.json")
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新 models-dir 路径
    config["models-dir"] = f"{mineru_path}/models/mineru"
    config["layoutreader-model-dir"] = f"{mineru_path}/models/mineru"
    
    # 保存更新后的配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 复制到用户目录
    user_home = os.path.expanduser("~")
    dest_file = os.path.join(user_home, "magic-pdf.json")
    shutil.copy2(config_file, dest_file)
    print(f"配置文件已更新并复制到: {dest_file}")
else:
    print(f"配置文件不存在: {config_file}")

