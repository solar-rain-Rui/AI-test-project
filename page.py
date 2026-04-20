#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
AI驱动pytest测试脚本生成工具
功能：输入需求 -> RAG检索接口 -> AI生成测试用例(JSON) -> 生成pytest脚本 -> 保存.py文件
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
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

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


def extract_all_jsons(text: str) -> list:
    import re as regex_module
    import json
    
    if not text:
        return []
    
    text = text.strip()
    
    def _extract_candidates(source_text):
        candidates = []
        
        brace_depth = 0
        json_start = -1
        for i, char in enumerate(source_text):
            if char == '{':
                if brace_depth == 0:
                    json_start = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and json_start != -1:
                    candidates.append((json_start, i + 1, source_text[json_start:i + 1]))
                    json_start = -1
        
        bracket_depth = 0
        array_start = -1
        for i, char in enumerate(source_text):
            if char == '[':
                if bracket_depth == 0:
                    array_start = i
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
                if bracket_depth == 0 and array_start != -1:
                    candidates.append((array_start, i + 1, source_text[array_start:i + 1]))
                    array_start = -1
        
        return candidates
    
    all_candidates = _extract_candidates(text)
    
    code_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in regex_module.finditer(code_pattern, text):
        code_content = match.group(1).strip()
        code_offset = match.start()
        for start, end, content in _extract_candidates(code_content):
            all_candidates.append((code_offset + start, code_offset + end, content))
    
    filtered = []
    for i, (s1, e1, c1) in enumerate(all_candidates):
        contained = False
        for j, (s2, e2, c2) in enumerate(all_candidates):
            if i != j and s2 <= s1 and e2 >= e1 and (s2 < s1 or e2 > e1):
                contained = True
                break
        if not contained:
            filtered.append((s1, e1, c1))
    
    seen = set()
    valid_jsons = []
    for start, end, candidate in filtered:
        try:
            candidate = regex_module.sub(r'"[^"]*"\.repeat\(\d+\)', '"repeated_string"', candidate)
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and not parsed:
                continue
            if isinstance(parsed, list) and not parsed:
                continue
            key = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                valid_jsons.append(parsed)
        except:
            continue
    
    return valid_jsons


def flatten_cases(data) -> list:
    if isinstance(data, list):
        cases = []
        for item in data:
            if isinstance(item, dict):
                if all(k in item for k in ["name", "input", "expected"]):
                    cases.append(item)
        return cases
    
    if isinstance(data, dict):
        cases = []
        for category in ["normal", "abnormal", "boundary"]:
            if category in data and isinstance(data[category], list):
                for item in data[category]:
                    if isinstance(item, dict):
                        case_copy = dict(item)
                        case_copy["_category"] = category
                        cases.append(case_copy)
        for key, value in data.items():
            if key not in ["normal", "abnormal", "boundary"]:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            if all(k in item for k in ["name", "input", "expected"]):
                                cases.append(item)
        return cases
    
    return []


def merge_all_cases(json_list: list) -> dict:
    all_cases = []
    for json_obj in json_list:
        cases = flatten_cases(json_obj)
        all_cases.extend(cases)
    
    result = {"normal": [], "abnormal": [], "boundary": []}
    uncategorized = []
    
    for case in all_cases:
        category = case.pop("_category", None)
        if category in ["normal", "abnormal", "boundary"]:
            result[category].append(case)
        else:
            uncategorized.append(case)
    
    if uncategorized:
        result["normal"].extend(uncategorized)
    
    return result


def extract_json(text: str) -> str:
    import json
    json_list = extract_all_jsons(text)
    if not json_list:
        return ""
    merged = merge_all_cases(json_list)
    return json.dumps(merged, ensure_ascii=False)


def extract_json_from_response(response_text):
    import re as regex_module
    
    if not response_text:
        print("[JSON解析失败] 输入内容为空")
        return None
    
    json_list = extract_all_jsons(response_text)
    
    if not json_list:
        print("[JSON解析失败] 未找到有效JSON内容")
        print(f"[原始输出(前1000字符)]: {response_text[:1000]}")
        return None
    
    print(f"[JSON解析] 找到 {len(json_list)} 个有效JSON块")
    
    result = merge_all_cases(json_list)
    
    total_cases = len(result.get("normal", [])) + len(result.get("abnormal", [])) + len(result.get("boundary", []))
    print(f"[JSON解析] 合并后共 {total_cases} 个测试用例")
    print(f"[调试] 处理后的JSON内容(前1000字符):\n{json.dumps(result, ensure_ascii=False)[:1000]}")
    
    if not result or all(len(result.get(k, [])) == 0 for k in ["normal", "abnormal", "boundary"]):
        print("[JSON解析结果] 空结果，无有效用例")
        return {}
    
    return result


async def generate_testcases(task):
    writer_message = """你是测试用例生成器。你必须且只能输出一个合法的JSON对象。
【绝对禁止】

* 禁止输出任何解释、说明、注释
* 禁止使用代码块标记 `json 或 `
* 禁止使用任何代码表达式
* 禁止输出伪代码或占位符
* 禁止编造接口路径、参数、返回字段

---

【必须遵守】

1. 输出必须是单个JSON对象，无任何其他内容

2. 每个用例必须包含以下字段：

* name（用例名称）
* path（接口路径）
* method（HTTP方法，如GET/POST）
* input（请求参数，必须是对象）
* expected（结构化断言对象）

---

3. input要求：

* 必须严格来自接口参数定义
* 不允许新增不存在的字段

---

4. expected要求（必须是结构化对象）：

格式如下：

{
"status_code": 200,
"body": {
"字段名": "期望值"
}
}

说明：

* status_code为HTTP状态码
* body中只包含需要校验的关键字段
* 不允许使用“成功/失败”等模糊描述

---

5. 必须严格基于提供的接口信息：

* path必须与接口信息一致
* method必须正确
* 参数必须来自接口定义
* 返回字段必须来自接口定义

---

6. 如果接口信息不足以生成完整用例，请输出空JSON对象 {}

---

---

【输出示例】

{
"normal": [
{
"name": "正常注册",
"path": "/register",
"method": "POST",
"input": {
"username": "testuser",
"password": "Pass1234"
},
"expected": {
"status_code": 200,
"body": {
"code": 0,
"msg": "success"
}
}
}
],
"abnormal": [],
"boundary": []
}

---

现在请根据需求和提供的接口信息生成测试用例JSON：
"""
    
    reviewer_message = """你是测试用例评审专家。

---

【重要前提】

如果当前没有提供接口定义信息（如Swagger、接口参数说明、返回结构等）：

- 跳过“接口一致性校验”
- 只进行结构校验和基本合理性校验

---

【评审标准】

1. 用例结构校验：

* 必须包含：
  name、path、method、input、expected

2. expected结构：

* 必须包含：
  status_code（整数）
  body（对象）

3. 基本合理性：

* 不存在明显逻辑错误（如空参数却期望成功）

---

【仅在提供接口定义时才执行】

4. 接口一致性校验：

* path必须存在
* method必须正确
* 参数必须来自接口定义
* 返回字段必须来自接口定义

---

【判定规则】

* 缺少结构字段 → REJECT
* expected不是结构化 → REJECT
* 否则 → APPROVE

---

注意：
如果没有接口定义，不允许因为“无法校验”而REJECT

"""
    
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
            chunk_type = getattr(chunk, 'type', None)
            content = getattr(chunk, 'content', '')
            
            if chunk_type == 'ModelClientStreamingChunkEvent':
                full_response += content
            elif hasattr(chunk, 'messages'):
                messages = getattr(chunk, 'messages', [])
                for msg in messages:
                    if hasattr(msg, 'content'):
                        msg_content = getattr(msg, 'content', '')
                        if msg_content and isinstance(msg_content, str):
                            full_response += msg_content
                            if "APPROVE" in msg_content or "REJECT" in msg_content:
                                review_result = msg_content
            elif content and isinstance(content, str):
                full_response += content
        elif isinstance(chunk, str):
            full_response += chunk
    
    print("\n" + "="*50)
    print("【LLM原始输出】")
    print(full_response[:2000] if len(full_response) > 2000 else full_response)
    if len(full_response) > 2000:
        print(f"... (共{len(full_response)}字符，已截断)")
    print("="*50)
    
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


def get_embedding(text: str) -> list: #将文本转为向量
    url = conf['deepseek']['base_url'].rstrip('/')
    if not url.endswith('/embeddings'):
        url = url.rsplit('/', 1)[0] + '/embeddings'
    headers = {
        "Authorization": f"Bearer {conf['deepseek']['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-v3",
        "input": text
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"[Embedding失败] {e}")
        return None

#解析Swagger Json，生成接口文本列表
def parse_swagger_to_docs(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        swagger = json.load(f)
    
    docs = []
    paths = swagger.get("paths", {})
    
    for path, methods in paths.items():
        for method, detail in methods.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue
            
            name = detail.get("summary", detail.get("operationId", path))
            
            params = []
            parameters = detail.get("parameters", [])
            for p in parameters:
                param_name = p.get("name", "")
                param_in = p.get("in", "")
                params.append(f"{param_name}({param_in})")
            
            request_body = detail.get("requestBody", {})
            if request_body:
                content = request_body.get("content", {})
                for content_type, schema_info in content.items():
                    schema = schema_info.get("schema", {})
                    properties = schema.get("properties", {})
                    for prop_name in properties:
                        params.append(f"{prop_name}(body)")
            
            responses = detail.get("responses", {})
            resp_fields = []
            for status_code, resp in responses.items():
                resp_content = resp.get("content", {})
                for ct, schema_info in resp_content.items():
                    schema = schema_info.get("schema", {})
                    props = schema.get("properties", {})
                    for prop_name in props:
                        resp_fields.append(prop_name)
            
            text = f"接口: {name} | 路径: {path} | 方法: {method.upper()} | 参数: {','.join(params) if params else '无'} | 返回: {','.join(resp_fields) if resp_fields else '无'}"
            
            docs.append({
                "name": name,
                "path": path,
                "method": method.upper(),
                "params": params,
                "response_fields": resp_fields,
                "text": text
            })
    
    print(f"[Swagger解析] 共解析 {len(docs)} 个接口")
    return docs

#构建FAISS向量索引
def build_faiss_index(docs: list):
    if faiss is None:
        print("[FAISS未安装] 跳过向量索引构建")
        return None, docs
    
    texts = [doc["text"] for doc in docs]
    embeddings = []
    
    for i, text in enumerate(texts):
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[Embedding跳过] 第{i+1}个接口向量化失败")
    
    if not embeddings:
        print("[FAISS构建失败] 无有效向量")
        return None, docs
    
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    
    valid_docs = [docs[i] for i in range(len(docs)) if i < len(embeddings)]
    
    print(f"[FAISS索引构建] 维度: {dimension}, 接口数: {len(valid_docs)}")
    return index, valid_docs

#相似度检索，返回最相关接口
def retrieve_api(task: str, index, docs, top_k=1):
    if index is None or faiss is None:
        print("[RAG检索跳过] FAISS索引不可用")
        return None
    
    query_emb = get_embedding(task)
    if query_emb is None:
        print("[RAG检索失败] 查询向量化失败")
        return None
    
    query_vector = np.array([query_emb], dtype=np.float32)
    distances, indices = index.search(query_vector, min(top_k, len(docs)))
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(docs):
            doc = docs[idx]
            doc["score"] = float(distances[0][i])
            results.append(doc)
            print(f"[RAG检索] 匹配接口: {doc['name']} | 路径: {doc['path']} | 距离: {doc['score']:.4f}")
    
    return results[0] if results else None

#构造增强Prompt，拼接接口信息
def build_rag_prompt(task: str, api_info: dict) -> str:
    if api_info is None:
        return task
    
    params_str = ", ".join(api_info.get("params", [])) if api_info.get("params") else "无"
    resp_str = ", ".join(api_info.get("response_fields", [])) if api_info.get("response_fields") else "无"
    
    rag_context = f"""
【接口信息（来自Swagger文档）】
接口名称: {api_info.get('name', '未知')}
请求路径: {api_info.get('path', '未知')}
请求方法: {api_info.get('method', '未知')}
请求参数: {params_str}
返回字段: {resp_str}

请基于以上接口信息生成测试用例，确保：
1. 测试用例的input参数与接口参数一致
2. 覆盖正常、异常、边界三种场景
3. 输出纯JSON格式
"""
    return rag_context + "\n原始需求: " + task


def validate_case(testcases):
    if testcases is None:
        return {
            "valid": False,
            "errors": [{"type": "structure_error", "message": "测试用例为空(None)"}]
        }
    
    if not isinstance(testcases, dict):
        return {
            "valid": False,
            "errors": [{"type": "structure_error", "message": "测试用例格式错误，应为字典"}]
        }
    
    if not testcases:
        return {
            "valid": False,
            "errors": [{"type": "empty_result", "message": "空JSON对象 {}，无有效用例"}]
        }
    
    errors = []
    has_valid_list = False
    
    for key, value in testcases.items():
        if isinstance(value, list) and len(value) > 0:
            has_valid_list = True
    
    if not has_valid_list:
        errors.append({
            "type": "no_valid_cases",
            "message": "没有任何非空用例列表，至少需要一个包含用例的分类"
        })
    
    known_categories = ["normal", "abnormal", "boundary"]
    for key, value in testcases.items():
        if key in known_categories:
            if not isinstance(value, list):
                errors.append({
                    "type": "type_error",
                    "message": f"{key}场景的用例应该是列表格式"
                })
                continue
            
            for idx, case in enumerate(value):
                if not isinstance(case, dict):
                    errors.append({
                        "type": "type_error",
                        "message": f"{key}场景第{idx+1}条用例格式错误"
                    })
                    continue
                
                required_fields = ["name", "input", "expected"]
                missing = [f for f in required_fields if f not in case]
                if missing:
                    errors.append({
                        "type": "missing_field",
                        "message": f"{key}场景第{idx+1}条用例缺少字段: {', '.join(missing)}"
                    })
                
                input_data = case.get("input", {})
                if not isinstance(input_data, dict):
                    errors.append({
                        "type": "type_error",
                        "message": f"{key}场景第{idx+1}条用例的input应该是对象格式"
                    })
                elif not input_data:
                    errors.append({
                        "type": "empty_field",
                        "message": f"{key}场景第{idx+1}条用例的input为空"
                    })
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }



def build_feedback_prompt(task: str, errors: list) -> str:
    error_messages = []
    for error in errors:
        error_messages.append(f"- {error['message']}")
    
    feedback = f"""你需要修复上一轮生成的测试用例问题。

【错误信息】
{chr(10).join(error_messages)}

---

【强制要求】

1. 你必须重新生成完整的测试用例JSON
2. 输出必须是一个合法JSON对象（以{{开头，以}}结尾）
3. 禁止输出任何解释、说明、分析、提示语
4. 禁止输出"以下是修正结果"等文本
5. 每个用例必须包含字段：
   - name
   - path
   - method
   - input
   - expected

6. expected必须为结构化对象：
{{
  "status_code": 200,
  "body": {{}}
}}

7. 严格基于接口信息生成，不允许编造字段

---

❗如果你输出的不是JSON，将视为失败

---

原始需求：
{task}

请直接输出JSON：
"""
    return feedback


async def generate_with_feedback(task: str, max_retry: int = 2):
    original_task = task
    retry_count = 0
    last_testcases = None
    
    while retry_count < max_retry:
        print(f"\n{'='*50}")
        print(f"[第{retry_count + 1}轮生成] 开始生成测试用例...")
        print(f"{'='*50}")
        
        current_task = task
        if retry_count == 1:
            print("[第2轮] 使用强化约束prompt...")
            current_task = original_task + """

---

【重要提醒】请务必生成至少一个完整测试用例：

* 必须包含 path、method、input、expected
* expected必须为结构化对象
* 禁止返回空JSON {}
* 禁止输出任何解释文本

如果输出不是JSON，将视为失败
"""
        
        testcases = await generate_testcases(current_task)
        
        if testcases is None:
            print(f"[第{retry_count + 1}轮生成] ⚠️ 无法提取有效JSON，触发下一轮")
            retry_count += 1
            continue
        
        if isinstance(testcases, dict) and not testcases:
            print(f"[第{retry_count + 1}轮生成] ⚠️ 空JSON，触发下一轮")
            retry_count += 1
            continue
        
        total = sum(len(testcases.get(k, [])) for k in ["normal", "abnormal", "boundary"])
        print(f"[第{retry_count + 1}轮生成] 提取到 {total} 个测试用例")
        
        validation_result = validate_case(testcases)
        
        if validation_result["valid"]:
            print(f"[第{retry_count + 1}轮生成] ✅ 校验通过！")
            return testcases
        else:
            print(f"[第{retry_count + 1}轮生成] ❌ 校验失败，但不再重试（固定两轮机制）")
            last_testcases = testcases
            retry_count += 1
    
    print(f"\n{'='*50}")
    print(f"[最终结果] 两轮生成完成")
    if last_testcases:
        print(f"[返回结果] 最后一次生成的测试用例")
    else:
        print(f"[返回结果] 无有效测试用例")
    print(f"{'='*50}")
    return last_testcases


def generate_pytest_script(testcases, base_url):
    """将测试用例转换为pytest测试脚本"""
    script_lines = [
        '#!/usr/bin/env python',
        '# -*- coding: utf-8 -*-',
        '"""AI生成的pytest测试脚本"""',
        'import requests',
        'import pytest',
        '',
        f'BASE_URL = "{base_url}"',
        '',
    ]
    
    case_index = 0
    for category in ["normal", "abnormal", "boundary"]:
        cases = testcases.get(category, [])
        for case in cases:
            case_index += 1
            name = case.get("name", f"测试用例{case_index}")
            method = case.get("method", "POST").upper()
            path = case.get("path", "")
            input_data = case.get("input", {})
            expected = case.get("expected", {})
            expected_status = expected.get("status_code", 200)
            expected_body = expected.get("body", {})
            
            func_name = f"test_{category}_{case_index}"
            func_name = func_name.replace("-", "_").replace(" ", "_")
            
            url = f'{{BASE_URL}}{path}' if path else 'BASE_URL'
            
            script_lines.append(f'def {func_name}():')
            script_lines.append(f'    """{name}"""')
            script_lines.append(f'    url = {url}')
            
            input_json = json.dumps(input_data, ensure_ascii=False)
            script_lines.append(f'    payload = {input_json}')
            
            if method == "GET":
                script_lines.append(f'    response = requests.get(url, params=payload, timeout=10)')
            else:
                script_lines.append(f'    response = requests.{method.lower()}(url, json=payload, timeout=10)')
            
            script_lines.append(f'    assert response.status_code == {expected_status}')
            
            if expected_body:
                for k, v in expected_body.items():
                    script_lines.append(f'    assert response.json().get("{k}") == {json.dumps(v, ensure_ascii=False)}')
            
            script_lines.append('')
    
    return '\n'.join(script_lines)


def main():
    st.title("🤖 AI驱动pytest测试脚本生成工具")
    st.markdown("**测试闭环**: 输入需求 → RAG检索接口 → AI生成测试用例(JSON) → 生成pytest脚本 → 保存.py文件")
    st.divider()
    
    user_input = st.text_area(
        "📝 请输入需求描述",
        height=150,
        placeholder="例如：用户注册接口，需要用户名、密码、邮箱三个字段..."
    )
    
    base_url = st.text_input(
        "🔗 接口Base URL",
        value="https://petstore.swagger.io/v2",
        help="接口基础URL，用于拼接完整请求路径"
    )
    
    swagger_path = st.text_input(
        "📄 Swagger文件路径（可选）",
        value="",
        help="上传Swagger JSON文件路径，启用RAG接口匹配"
    )
    
    if st.button("🚀 生成pytest测试脚本", type="primary"):
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
            if swagger_path and os.path.exists(swagger_path):
                with st.spinner("🔍 RAG检索匹配接口..."):
                    try:
                        docs = parse_swagger_to_docs(swagger_path)
                        index, docs = build_faiss_index(docs)
                        api_info = retrieve_api(task, index, docs)
                        if api_info:
                            task = build_rag_prompt(task, api_info)
                            st.info(f"📌 RAG匹配接口: **{api_info['name']}** ({api_info['method']} {api_info['path']})")
                        else:
                            st.warning("RAG未匹配到相关接口，使用原始需求生成")
                    except Exception as e:
                        st.warning(f"RAG检索失败: {str(e)}，使用原始需求生成")
            
            try:
                testcases = asyncio.run(generate_with_feedback(task))
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
        
        st.subheader("🐍 生成的pytest测试脚本")
        pytest_script = generate_pytest_script(testcases, base_url)
        
        st.code(pytest_script, language="python")
        
        script_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_generated.py')
        with open(script_file_path, "w", encoding="utf-8") as f:
            f.write(pytest_script)
        st.success(f"✅ pytest测试脚本已保存到: test_generated.py")
        
        st.subheader("▶️ 运行测试")
        st.code("pytest test_generated.py -v", language="bash")


if __name__ == "__main__":
    main()
