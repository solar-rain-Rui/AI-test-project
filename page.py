#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
AI辅助接口自动化测试工具（简化版）
功能：输入需求 -> AI生成测试用例(JSON) -> 执行接口测试 -> 输出结果
"""
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from configparser import ConfigParser
import streamlit as st
import asyncio
import requests
import json
import re
import os

st.set_page_config(
    page_title="AI辅助接口自动化测试工具",
    page_icon="🤖",
    layout="centered"
)

conf = ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
conf.read(config_path)

model_info = {
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


def get_testcase_writer(model_client, system_message):
    return AssistantAgent(
        name="testcase_writer",
        model_client=model_client,
        system_message=system_message,
    )


def extract_json_from_response(response_text):
    json_str = response_text
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    if match:
        json_str = match.group(1)
    else:
        brace_pattern = r'\{[\s\S]*\}'
        brace_match = re.search(brace_pattern, response_text)
        if brace_match:
            json_str = brace_match.group(0)
    try:
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        print(f"[JSON解析失败] 错误信息: {e}")
        print(f"[原始输出内容]:\n{response_text}")
        return None


async def generate_testcases(task):
    system_message = """你是一位资深测试工程师，请根据需求生成接口测试用例。
必须严格按照以下JSON格式输出，不要输出任何其他内容：
{
    "normal": [
        {"name": "用例名称", "input": {"key": "value"}, "expected": "预期结果"}
    ],
    "abnormal": [
        {"name": "用例名称", "input": {"key": "value"}, "expected": "预期结果"}
    ],
    "boundary": [
        {"name": "用例名称", "input": {"key": "value"}, "expected": "预期结果"}
    ]
}
"""
    model_client = OpenAIChatCompletionClient(
        model=conf['deepseek']['model'],
        base_url=conf['deepseek']['base_url'],
        api_key=conf['deepseek']['api_key'],
        model_info=model_info,
    )
    testcase_writer = get_testcase_writer(model_client, system_message)
    termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        participants=[testcase_writer],
        termination_condition=termination,
        max_turns=1
    )
    full_response = ""
    async for chunk in team.run_stream(task=task):
        if hasattr(chunk, 'content') and hasattr(chunk, 'type'):
            if chunk.type != 'ModelClientStreamingChunkEvent':
                full_response += chunk.content
        elif isinstance(chunk, str):
            full_response += chunk
    return extract_json_from_response(full_response)


def execute_api_test(test_url, testcases_dict):
    results = []
    for category in ["normal", "abnormal", "boundary"]:
        cases = testcases_dict.get(category, [])
        for case in cases:
            case_name = case.get("name", "未命名用例")
            input_data = case.get("input", {})
            expected = case.get("expected", "")
            try:
                response = requests.post(test_url, json=input_data, timeout=10)
                status = "PASS" if response.status_code == 200 else "FAIL"
                actual_status = response.status_code
            except Exception as e:
                status = "ERROR"
                actual_status = str(e)
            results.append({
                "category": category,
                "name": case_name,
                "input": input_data,
                "expected": expected,
                "actual_status": actual_status,
                "result": status
            })
    return results


def main():
    st.title("🤖 AI辅助接口自动化测试工具")
    st.markdown("**测试闭环**: 输入需求 → AI生成测试用例(JSON) → 执行接口测试 → 输出结果")
    st.divider()
    
    user_input = st.text_area(
        "📝 请输入需求描述",
        height=150,
        placeholder="例如：用户注册接口，需要用户名、密码、邮箱三个字段..."
    )
    
    test_url = st.text_input(
        "🔗 测试接口URL",
        value="https://httpbin.org/post",
        help="用于执行接口测试的目标URL"
    )
    
    if st.button("🚀 生成并执行测试", type="primary"):
        if not user_input:
            st.error("请输入需求描述！")
            return
        
        if not conf['deepseek']['api_key']:
            st.error("请先在config.ini中配置DeepSeek的API Key！")
            return
        
        with st.spinner("AI正在生成测试用例..."):
            task = f"""
需求描述: {user_input}

请生成接口测试用例，包含正常场景、异常场景和边界场景。
必须输出JSON格式，不要输出任何其他内容。
"""
            try:
                testcases = asyncio.run(generate_testcases(task))
            except Exception as e:
                st.error(f"AI生成测试用例失败: {str(e)}")
                return
        
        if testcases is None:
            st.error("AI返回的内容无法解析为JSON，请查看控制台输出！")
            return
        
        json_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data.json')
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(testcases, f, ensure_ascii=False, indent=2)
        st.info(f"✅ 测试用例已保存到: test_data.json")
        
        st.subheader("📋 AI生成的测试用例(JSON)")
        st.json(testcases)
        
        st.subheader("🧪 接口测试执行结果")
        with st.spinner("正在执行接口测试..."):
            results = execute_api_test(test_url, testcases)
        
        pass_count = sum(1 for r in results if r["result"] == "PASS")
        fail_count = sum(1 for r in results if r["result"] == "FAIL")
        error_count = sum(1 for r in results if r["result"] == "ERROR")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ 通过", pass_count)
        col2.metric("❌ 失败", fail_count)
        col3.metric("⚠️ 错误", error_count)
        
        st.divider()
        for r in results:
            with st.expander(f"[{r['category']}] {r['name']} - {r['result']}", expanded=False):
                st.write(f"**输入数据**: {r['input']}")
                st.write(f"**预期结果**: {r['expected']}")
                st.write(f"**实际状态**: {r['actual_status']}")
                st.write(f"**测试结果**: {r['result']}")


if __name__ == "__main__":
    main()
