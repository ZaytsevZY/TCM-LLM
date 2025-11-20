#!/bin/bash
# ==============================================================================
# 使用 ModelScope 下载 Qwen2.5-7B-Instruct 模型
# 适用于国内网络环境，速度快且稳定
# ==============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Qwen2.5-7B-Instruct 模型下载工具${NC}"
echo -e "${GREEN}  使用 ModelScope 国内镜像${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "${YELLOW}📁 项目目录: ${PROJECT_ROOT}${NC}"

# 模型保存路径
MODEL_CACHE_DIR="${HOME}/.cache/modelscope/hub"
echo -e "${YELLOW}💾 模型缓存目录: ${MODEL_CACHE_DIR}${NC}"
echo ""

# 步骤1: 清理旧文件
echo -e "${YELLOW}🗑️  步骤 1/4: 清理旧的下载文件...${NC}"
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
rm -rf ~/.cache/huggingface/hub/.locks/models--Qwen--Qwen2.5-7B-Instruct/
rm -rf ~/TCM-LLM/models/Qwen*
echo -e "${GREEN}✅ 清理完成${NC}"
echo ""

# 步骤2: 安装依赖
echo -e "${YELLOW}📦 步骤 2/4: 安装 ModelScope...${NC}"
pip install modelscope -U
echo -e "${GREEN}✅ ModelScope 安装完成${NC}"
echo ""

# 步骤3: 下载模型
echo -e "${YELLOW}⬇️  步骤 3/4: 开始下载模型...${NC}"
echo -e "${YELLOW}   模型大小: ~15GB${NC}"
echo -e "${YELLOW}   预计时间: 10-20分钟 (取决于网速)${NC}"
echo ""

python3 << 'PYTHON'
from modelscope import snapshot_download
import os
import sys

try:
    print("正在连接 ModelScope...")
    model_dir = snapshot_download(
        'Qwen/Qwen2.5-7B-Instruct',
        cache_dir=os.path.expanduser('~/.cache/modelscope/hub/')
    )
    
    print(f"\n✅ 模型下载完成！")
    print(f"📁 模型路径: {model_dir}")
    
    # 检查大小
    import subprocess
    result = subprocess.run(['du', '-sh', model_dir], capture_output=True, text=True)
    print(f"💾 模型大小: {result.stdout.strip().split()[0]}")
    
    # 保存路径到文件
    model_path_file = os.path.expanduser('~/TCM-LLM/.model_path')
    with open(model_path_file, 'w') as f:
        f.write(model_dir)
    print(f"\n📝 模型路径已保存到: {model_path_file}")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 下载失败，请检查网络连接${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ 步骤 3/4 完成${NC}"
echo ""

# 步骤4: 更新训练脚本
echo -e "${YELLOW}🔧 步骤 4/4: 更新训练脚本配置...${NC}"

# 读取模型路径
MODEL_PATH=$(cat ~/TCM-LLM/.model_path)

# 备份原始训练脚本
if [ ! -f "${PROJECT_ROOT}/scripts/04_train.sh.backup" ]; then
    cp "${PROJECT_ROOT}/scripts/04_train.sh" "${PROJECT_ROOT}/scripts/04_train.sh.backup"
    echo -e "${GREEN}✅ 已备份原始训练脚本${NC}"
fi

# 更新模型路径
sed -i "s|--model_name_or_path Qwen/Qwen2.5-7B-Instruct|--model_name_or_path ${MODEL_PATH}|g" \
    "${PROJECT_ROOT}/scripts/04_train.sh"

echo -e "${GREEN}✅ 训练脚本已更新${NC}"
echo ""

# 完成
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🎉 模型下载并配置成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}📊 模型信息:${NC}"
echo -e "   路径: ${MODEL_PATH}"
echo -e "   大小: $(du -sh ${MODEL_PATH} | cut -f1)"
echo ""
echo -e "${YELLOW}🚀 下一步操作:${NC}"
echo -e "   运行训练: ${GREEN}bash scripts/04_train.sh${NC}"
echo ""
echo -e "${YELLOW}💡 提示:${NC}"
echo -e "   - 原始训练脚本已备份到: scripts/04_train.sh.backup"
echo -e "   - 如需恢复: cp scripts/04_train.sh.backup scripts/04_train.sh"
echo ""
