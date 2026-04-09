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
    import re as regex_module
    
    if not response_text:
        print("[JSON解析失败] 输入内容为空")
        return None
    
    json_str = response_text.strip()
    
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = regex_module.search(json_pattern, json_str)
    if match:
        json_str = match.group(1).strip()
    
    first_brace = json_str.find('{')
    last_brace = json_str.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = json_str[first_brace:last_brace + 1]
    
    def replace_repeat(match):
        try:
            char = match.group(1) if match.group(1) else 'a'
            count = int(match.group(2)) if match.group(2) else 10
            count = min(count, 100)
            return '"' + char * count + '"'
        except:
            return '"' + 'a' * 10 + '"'
    
    json_str = regex_module.sub(r'"([^"]*)"\s*\.repeat\s*\(\s*(\d+)\s*\)', replace_repeat, json_str)
    json_str = regex_module.sub(r'\.repeat\s*\(\s*(\d+)\s*\)', replace_repeat, json_str)
    json_str = regex_module.sub(r'"([a-zA-Z])"\s*\*\s*(\d+)', replace_repeat, json_str)
    
    json_str = regex_module.sub(r'[\u00a0\u3000]', ' ', json_str)
    json_str = regex_module.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
    
    json_str = regex_module.sub(r',\s*}', '}', json_str)
    json_str = regex_module.sub(r',\s*]', ']', json_str)
    json_str = regex_module.sub(r'}\s*{', '},{', json_str)
    
    json_str = regex_module.sub(r'"\s*:\s*"', '":"', json_str)
    json_str = regex_module.sub(r'"\s*,\s*"', '","', json_str)
    json_str = regex_module.sub(r'\[\s*\]', '[]', json_str)
    
    json_str = regex_module.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_str)
    
    print(f"[调试] 处理后的JSON内容(前1000字符):\n{json_str[:1000]}")
    
    try:
        result_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[JSON解析失败] 错误信息: {e}")
        print(f"[错误位置] 行: {e.lineno}, 列: {e.colno}")
        print(f"[错误附近内容]: ...{json_str[max(0, e.pos-50):e.pos+50]}...")
        return None
    
    required_keys = ["normal", "abnormal", "boundary"]
    missing_keys = [key for key in required_keys if key not in result_dict]
    if missing_keys:
        print(f"[JSON结构错误] 缺少必要字段: {', '.join(missing_keys)}")
        print(f"[当前字段]: {list(result_dict.keys())}")
        return None
    
    return result_dict


async def generate_testcases(task):
    writer_message = """你是测试用例生成器。你必须且只能输出一个合法的JSON对象。

【绝对禁止】
- 禁止输出任何解释、说明、注释
- 禁止使用代码块标记 ```json 或 ```
- 禁止使用任何代码表达式（如 .repeat()、函数调用等）
- 禁止在JSON字符串中使用换行
- 禁止输出伪代码或占位符

【必须遵守】
- 输出必须是单个JSON对象，无任何其他内容
- 所有字符串值必须是具体值（如 "testuser"、"Pass1234"）
- 每个用例必须包含三个字段：name、input、expected
- input必须是对象，不能是字符串

【输出示例】
{"normal":[{"name":"正常注册","input":{"username":"testuser","password":"Pass1234","email":"test@example.com"},"expected":"注册成功"}],"abnormal":[{"name":"用户名为空","input":{"username":"","password":"Pass1234","email":"test@example.com"},"expected":"注册失败"}],"boundary":[{"name":"用户名最小长度","input":{"username":"abc","password":"Pass1234","email":"test@example.com"},"expected":"注册成功"}]}

现在请根据需求生成测试用例JSON："""
    
    reviewer_message = """你是测试用例评审专家。你的职责是评审上一步生成的测试用例。

【评审标准】
1. 测试用例是否覆盖正常、异常、边界三类场景
2. 每条用例是否包含name、input、expected字段
3. input是否为有效对象
4. 用例是否符合业务逻辑（无明显的反常识问题）

【输出格式】
如果用例质量合格，输出：
APPROVE - 评审通过：[简短原因]

如果用例质量不合格，输出：
REJECT - 评审不通过：[具体问题说明]

注意：你只需要输出评审结果，不需要修改测试用例内容。"""
    
    model_client = OpenAIChatCompletionClient(
        model=conf['deepseek']['model'],
        base_url=conf['deepseek']['base_url'],
        api_key=conf['deepseek']['api_key'],
        model_info=model_info,
    )
    
    testcase_writer = AssistantAgent(
        name="testcase_writer",
        model_client=model_client,
        system_message=writer_message,
    )
    
    testcase_reviewer = AssistantAgent(
        name="testcase_reviewer",
        model_client=model_client,
        system_message=reviewer_message,
    )
    
    termination = TextMentionTermination("APPROVE")
    team = RoundRobinGroupChat(
        participants=[testcase_writer, testcase_reviewer],
        termination_condition=termination,
        max_turns=2
    )
    
    full_response = ""
    review_result = ""
    async for chunk in team.run_stream(task=task):
        if hasattr(chunk, 'content') and hasattr(chunk, 'type'):
            if chunk.type != 'ModelClientStreamingChunkEvent':
                content = chunk.content
                full_response += content
                if "APPROVE" in content or "REJECT" in content:
                    review_result = content
        elif isinstance(chunk, str):
            full_response += chunk
    
    print(f"[评审结果] {review_result}")
    
    return extract_json_from_response(full_response)


def validate_testcases(testcases):
    if not testcases or not isinstance(testcases, dict):
        return {"valid_testcases": {}, "stats": {"original": 0, "valid": 0, "filtered": 0}}
    
    valid_testcases = {"normal": [], "abnormal": [], "boundary": []}
    original_count = 0
    valid_count = 0
    filtered_count = 0
    
    for category in ["normal", "abnormal", "boundary"]:
        cases = testcases.get(category, [])
        for case in cases:
            original_count += 1
            
            if not isinstance(case, dict):
                filtered_count += 1
                continue
            
            if not all(key in case for key in ["name", "input", "expected"]):
                filtered_count += 1
                continue
            
            name = case.get("name", "")
            input_data = case.get("input", {})
            expected = case.get("expected", "")
            
            if not name or not name.strip():
                filtered_count += 1
                continue
            
            if not isinstance(input_data, dict) or not input_data:
                filtered_count += 1
                continue
            
            if not expected or not str(expected).strip():
                filtered_count += 1
                continue
            
            if category == "normal":
                if not input_data.get("username") or not input_data.get("password"):
                    if "成功" in str(expected):
                        filtered_count += 1
                        continue
            
            if category == "abnormal":
                if not input_data.get("username") and "成功" in str(expected):
                    filtered_count += 1
                    continue
            
            valid_testcases[category].append(case)
            valid_count += 1
    
    stats = {
        "original": original_count,
        "valid": valid_count,
        "filtered": filtered_count
    }
    
    print(f"[用例校验统计] 原始: {original_count}, 有效: {valid_count}, 过滤: {filtered_count}")
    
    return {"valid_testcases": valid_testcases, "stats": stats}


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
        
        validation_result = validate_testcases(testcases)
        valid_testcases = validation_result["valid_testcases"]
        stats = validation_result["stats"]
        
        st.subheader("🔍 用例校验统计")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📝 原始用例", stats["original"])
        col_b.metric("✅ 有效用例", stats["valid"])
        col_c.metric("🗑️ 过滤用例", stats["filtered"])
        
        if stats["valid"] == 0:
            st.error("校验后无有效用例，请检查AI生成结果！")
            return
        
        testcases = valid_testcases
        
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
