#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pip 镜像源测速工具
"""
import time
import urllib.request
import sys

# 要测试的镜像源
MIRRORS = {
    "清华大学": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "阿里云": "https://mirrors.aliyun.com/pypi/simple",
    "中科大": "https://pypi.mirrors.ustc.edu.cn/simple",
    "豆瓣": "https://pypi.douban.com/simple",
    "华为云": "https://mirrors.huaweicloud.com/repository/pypi/simple",
    "腾讯云": "https://mirrors.cloud.tencent.com/pypi/simple",
    "官方源": "https://pypi.org/simple"
}

# 测试文件（小包）
TEST_PACKAGE = "pip"

print("=" * 70)
print("🚀 PyPI 镜像源测速")
print("=" * 70)
print(f"测试包: {TEST_PACKAGE}")
print("-" * 70)

results = []

for name, mirror in MIRRORS.items():
    try:
        # 构建测试 URL
        url = f"{mirror}/{TEST_PACKAGE}/"
        
        # 测速
        print(f"⏱️  测试 {name:10s} ... ", end='', flush=True)
        start = time.time()
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'pip/21.0')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()
            elapsed = time.time() - start
            
        print(f"✓ {elapsed*1000:.0f} ms")
        results.append((name, mirror, elapsed))
        
    except Exception as e:
        print(f"✗ 失败 ({str(e)[:30]})")
        results.append((name, mirror, 999))

# 排序并显示结果
print("\n" + "=" * 70)
print("📊 测速结果（从快到慢）")
print("=" * 70)

results.sort(key=lambda x: x[2])

for i, (name, mirror, elapsed) in enumerate(results, 1):
    if elapsed < 999:
        speed = "🚀 极快" if elapsed < 0.3 else "⚡ 快速" if elapsed < 1 else "🐢 一般"
        print(f"{i}. {name:10s}  {elapsed*1000:6.0f} ms  {speed}")
        if i == 1:
            print(f"   推荐命令: pip install torch -i {mirror}")
    else:
        print(f"{i}. {name:10s}  连接失败")

print("\n" + "=" * 70)