#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
【LLM模型配置模块】
定义不同LLM提供商的模型参数配置
支持的模型: DeepSeek, Qwen
"""

model_deepseek_info = {
        "name": "deepseek-chat",
        "parameters": {
            "max_tokens": 2048,
            "temperature": 0.4,
            "top_p": 0.9
        },
        "family": "gpt-4o",
        "functions": [],
        "vision": False,
        "json_output": True,
        "function_calling": True,
        "structured_output": True
    }

model_qwen_info = {
        "name": "qwen-chat",
        "parameters": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8
        },
        "family": "gpt-4o",
        "functions": [],
        "vision": False,
        "json_output": True,
        "function_calling": True,
        "structured_output": True
    }
