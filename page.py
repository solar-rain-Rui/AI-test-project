#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
AI驱动测试用例生成平台
功能：输入需求 -> RAG检索接口 -> PromptTemplate构建 -> AI生成测试用例(JSON) -> 输出结构化JSON文件

【项目定位】
  本项目是"基于RAG + Multi-Agent的AI测试用例生成平台"，而非自动化执行平台。
  最终输出为结构化JSON测试用例文件，便于：
  1. 后续对接不同测试框架（pytest、JUnit、TestNG等）
  2. 人工审核和修改
  3. 测试用例管理系统导入
  4. 跨团队、跨语言复用

【为什么输出JSON而非pytest脚本】
  JSON是中间标准格式，具有以下优势：
  1. 框架无关性：不绑定特定测试框架，可转换为任意格式
  2. 可读性强：结构清晰，便于人工审核和修改
  3. 易于扩展：新增字段不影响现有解析逻辑
  4. 便于集成：可直接导入测试用例管理系统
  5. 支持多语言：Python/Java/JavaScript等均可解析

【JSON测试用例结构说明】
  {
    "title": "用例标题",
    "method": "HTTP方法(GET/POST/PUT/DELETE/PATCH)",
    "path": "接口路径(如/pet/{petId})",
    "params": {
      "path": {"petId": 123},      # 路径参数
      "query": {"status": "sold"}, # 查询参数
      "body": {"name": "test"}     # 请求体参数
    },
    "expected": {
      "status_code": 200,          # 期望HTTP状态码
      "body": {"code": 0}          # 期望响应体字段
    },
    "case_type": "正常场景/异常场景/边界场景"
  }

【模块架构】
├── RAG检索模块(rag/): chunker, embedder, retriever, reranker, pipeline
├── Prompt模板模块(prompt/): template
├── 用例生成模块: generate_testcases, generate_with_feedback
├── 校验模块: validate_testcases, validate_case
├── 后处理模块: postprocess_testcases
├── 质量评估模块(quality/): evaluator
└── JSON输出模块: generate_json_testcases, save_json_file
"""

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from configparser import ConfigParser
from rag.pipeline import RAGPipeline
from prompt.template import build_prompt, build_reviewer_prompt, build_retry_prompt, build_reject_retry_prompt
from quality.evaluator import evaluate_testcases
import streamlit as st
import asyncio
import requests
import json
import re
import os
import logging
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

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
    """
    创建测试用例生成Agent
    用于生成符合接口规范的测试用例JSON
    """
    return AssistantAgent(
        name="testcase_writer",
        model_client=model_client,
        system_message=system_message,
    )


def extract_all_jsons(text: str) -> list:
    """
    【JSON解析核心函数】从LLM输出文本中提取所有有效JSON块
    解决LLM输出不稳定问题，支持多种格式：
    1. 纯JSON对象/数组
    2. 代码块包裹的JSON (```json ... ```)
    3. 混合文本中的JSON片段
    """
    import re as regex_module
    import json
    
    if not text:
        return []
    
    text = text.strip()
    
    def _extract_candidates(source_text):
        """通过括号匹配提取候选JSON块"""
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
    """
    将嵌套的JSON结构扁平化为测试用例列表
    支持多种输入格式：
    - 列表格式: [{"name": ..., "input": ..., "expected": ...}]
    - 分类格式: {"normal": [...], "abnormal": [...], "boundary": [...]}
    """
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
    """
    合并多个JSON块的测试用例，按场景分类
    输出格式: {"normal": [...], "abnormal": [...], "boundary": [...]}
    """
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
    """提取并合并JSON，返回JSON字符串"""
    import json
    json_list = extract_all_jsons(text)
    if not json_list:
        return ""
    merged = merge_all_cases(json_list)
    return json.dumps(merged, ensure_ascii=False)


def extract_json_from_response(response_text):
    """
    【LLM输出解析入口】从LLM响应中提取并合并测试用例
    处理流程: 提取JSON块 -> 扁平化 -> 合并分类 -> 返回结构化用例
    """
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


async def generate_testcases(task, base_prompt=""):
    """
    【核心生成函数】多Agent协作生成测试用例
    架构: testcase_writer(生成) <-> testcase_reviewer(评审)
    流程: Writer生成JSON -> Reviewer评审 -> APPROVE终止/REJECT继续

    为什么使用base_prompt而非rag_context:
      base_prompt是build_prompt构建的完整基础Prompt，包含:
      1. ROLE_DEFINITION - 角色定义
      2. RAG上下文 - 接口定义信息
      3. TASK_DESCRIPTION - 任务说明
      4. OUTPUT_FORMAT - 输出格式约束
      5. OUTPUT_EXAMPLE - 输出示例
      使用base_prompt作为Writer的system_message，确保Writer始终基于完整Prompt结构工作。
      REJECT retry和JSON校验retry时，也基于同一份base_prompt追加约束，不丢失原始结构。

    为什么不能覆盖增强后的Prompt:
      之前的实现中，task变量在多处被重新赋值，导致RAG上下文丢失。
      现在base_prompt作为独立参数传入，与task分离维护，
      确保基础Prompt在整个Agent协作流程中始终存在，retry只追加约束。

    输入:
      task: 用户需求（作为user message传给Writer）
      base_prompt: 基础Prompt（来自build_prompt，作为system_message）
    """
    writer_message = base_prompt if base_prompt else task
    
    rag_context_for_reviewer = ""
    if base_prompt and "接口信息" in base_prompt:
        rag_context_for_reviewer = base_prompt
    
    reviewer_message = build_reviewer_prompt(rag_context_for_reviewer)
    
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
    
    """
    【AutoGen 0.4.x API变化说明】
    
    为什么generate_reply不可用:
      AutoGen 0.4.x版本进行了重大架构重构，采用异步事件驱动架构。
      旧版0.2.x的generate_reply方法已被移除，不再支持。
    
    当前版本AutoGen正确调用方式:
      1. 使用on_messages()方法替代generate_reply()
      2. on_messages是异步方法，需要await
      3. 需要传入TextMessage列表和CancellationToken
      4. Agent是有状态的，自动维护对话历史
      5. 每次调用只需传入新消息，不需要传入完整历史
      6. 返回Response对象，通过response.chat_message.content获取内容
    
    示例:
      cancellation_token = CancellationToken()
      response = await agent.on_messages(
          [TextMessage(content="任务内容", source="user")],
          cancellation_token
      )
      content = response.chat_message.content
    """
    
    cancellation_token = CancellationToken()
    
    max_turns = 2
    current_turn = 1
    
    full_response = ""
    review_result = ""
    last_writer_reply = ""
    
    while current_turn <= max_turns:
        print(f"\n[第{current_turn}轮] Writer生成测试用例...")
        
        writer_response = await testcase_writer.on_messages(
            [TextMessage(content=task, source="user")],
            cancellation_token
        )
        
        writer_reply = writer_response.chat_message.content
        last_writer_reply = writer_reply
        full_response += writer_reply
        
        print(f"[第{current_turn}轮] Reviewer评审中...")
        
        reviewer_content = f"请评审以下测试用例：\n{writer_reply}"
        
        reviewer_response = await testcase_reviewer.on_messages(
            [TextMessage(content=reviewer_content, source="user")],
            cancellation_token
        )
        reviewer_reply = reviewer_response.chat_message.content
        full_response += "\n" + reviewer_reply
        
        if "APPROVE" in reviewer_reply:
            review_result = reviewer_reply
            print(f"[第{current_turn}轮] ✅ APPROVE - 评审通过")
            break
        elif "REJECT" in reviewer_reply:
            review_result = reviewer_reply
            print(f"[第{current_turn}轮] ❌ REJECT - 需要修正")
            
            if current_turn < max_turns:
                task = build_reject_retry_prompt(base_prompt, reviewer_reply)
        else:
            review_result = reviewer_reply
            print(f"[第{current_turn}轮] ⚠️ 未明确判定，结束流程")
            break
        
        current_turn += 1
    
    print("\n" + "="*50)
    print("【LLM原始输出】")
    print(full_response[:2000] if len(full_response) > 2000 else full_response)
    if len(full_response) > 2000:
        print(f"... (共{len(full_response)}字符，已截断)")
    print("="*50)
    
    print(f"[评审结果] {review_result}")
    
    return extract_json_from_response(full_response)


def validate_testcases(testcases):
    """
    【用例校验函数】验证测试用例的结构完整性和逻辑合理性
    校验项: 必填字段、input非空、expected有效性
    返回: 有效用例 + 统计信息
    """
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


def get_embedding(text: str) -> list:
    """
    【RAG模块】调用DeepSeek Embedding API将文本转为向量
    用于构建FAISS索引和语义检索
    """
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
        response = requests.post(url, headers=headers, json=payload, timeout=30)#向 DeepSeek 的 Embedding 服务发送请求，超时时间 30 秒。
        result = response.json()
        return result["data"][0]["embedding"] #返回全部的向量，一堆浮点数的列表
    except Exception as e:
        print(f"[Embedding失败] {e}")
        return None

#解析Swagger Json，生成接口文本列表
def get_swagger_base_url(swagger):
    """
    【RAG模块】从Swagger解析base_url
    支持OpenAPI 3.0 (servers字段) 和 Swagger 2.0 (host + basePath)
    """
    servers = swagger.get("servers", [])
    if servers and isinstance(servers, list):
        first_server = servers[0]
        if isinstance(first_server, dict):
            url = first_server.get("url")
            if url and url.startswith("http"):
                return url
    
    host = swagger.get("host")
    if host:
        schemes = swagger.get("schemes", ["https"])
        scheme = schemes[0] if isinstance(schemes, list) and schemes else "https"
        base_path = swagger.get("basePath", "")
        url = f"{scheme}://{host}{base_path}"
        if url.startswith("http"):
            return url
    
    return None


def parse_swagger_to_docs(file_path: str):
    """
    【RAG模块】解析Swagger JSON文件，提取接口信息
    输出: 接口文档列表 + base_url
    每个接口包含: name, path, method, params, response_fields, text(用于向量化)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        swagger = json.load(f) #把json文件解析成字典/列表
    
    base_url = get_swagger_base_url(swagger) #提取出url
    
    docs = []
    paths = swagger.get("paths", {})
    
    for path, methods in paths.items():#循环遍历
        for method, detail in methods.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue #只处理常见的5中http方法
            
            name = detail.get("summary", detail.get("operationId", path))
            #确定接口可读名称  优先级：summary(中文描述) > operationId(函数名) > path(路径名)
            params = []
            parameters = detail.get("parameters", []) #获取路径参数，查询参数等
            for p in parameters:
                param_name = p.get("name", "")
                param_in = p.get("in", "")
                params.append(f"{param_name}({param_in})")
            
            request_body = detail.get("requestBody", {})#处理POST/PUT 等方法特有的 requestBody（请求体）
            if request_body:
                content = request_body.get("content", {})
                for content_type, schema_info in content.items():
                    schema = schema_info.get("schema", {})
                    properties = schema.get("properties", {})
                    for prop_name in properties:# 遍历请求体里的每一个字段名，并标注为 (body)
                        params.append(f"{prop_name}(body)")
            
            responses = detail.get("responses", {})# 处理返回结果（Responses）
            resp_fields = []
            for status_code, resp in responses.items(): #遍历状态码和对应响应内容
                resp_content = resp.get("content", {})
                for ct, schema_info in resp_content.items():
                    schema = schema_info.get("schema", {})
                    props = schema.get("properties", {})
                    for prop_name in props:# 提取返回结果中定义的字段名
                        resp_fields.append(prop_name)
            
            text = f"接口: {name} | 路径: {path} | 方法: {method.upper()} | 参数: {','.join(params) if params else '无'} | 返回: {','.join(resp_fields) if resp_fields else '无'}"
            #把每个接口的所有信息打包成一个字典，存进 docs 列表里
            docs.append({
                "name": name,
                "path": path,
                "method": method.upper(),
                "params": params,
                "response_fields": resp_fields,
                "text": text
            })
    
    print(f"[Swagger解析] 共解析 {len(docs)} 个接口")
    if base_url:
        print(f"[Swagger解析] base_url: {base_url}")
    return docs, base_url

#构建FAISS向量索引
def build_faiss_index(docs: list):
    """
    【RAG模块】构建FAISS向量索引
    流程: 接口文本 -> Embedding向量化 -> FAISS L2索引
    用于后续语义检索匹配最相关接口
    """
    if faiss is None:
        print("[FAISS未安装] 跳过向量索引构建")
        return None, docs
    
    texts = [doc["text"] for doc in docs] #拿出每个接口的描述文本
    embeddings = []
    #遍历文本并生成向量
    for i, text in enumerate(texts):
        emb = get_embedding(text) #把接口文本转换成向量，让语义相似的接口在向量空间里更接近
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[Embedding跳过] 第{i+1}个接口向量化失败")
    
    if not embeddings:
        print("[FAISS构建失败] 无有效向量")
        return None, docs
    
    embedding_matrix = np.array(embeddings, dtype=np.float32) #转换为numpy矩阵(构建向量矩阵)
    
    dimension = embedding_matrix.shape[1] #获取向量的维度
    index = faiss.IndexFlatL2(dimension)#创建 FAISS 索引，使用 L2 距离（欧式距离） 来衡量向量之间相似度
    index.add(embedding_matrix) #把所有接口向量存进索引里，构建可检索的数据结构
    #FAISS索引 ≈ 轻量向量数据库（功能上）它既存向量，又负责检索
    valid_docs = [docs[i] for i in range(len(docs)) if i < len(embeddings)]
    
    print(f"[FAISS索引构建] 维度: {dimension}, 接口数: {len(valid_docs)}")
    return index, valid_docs #返回索引和对应接口信息，用于后续RAG检索最相关接口

#相似度检索，返回最相关接口
def retrieve_api(task: str, index, docs, top_k=1):
    """
    【RAG模块】语义检索匹配最相关的接口
    流程: 需求描述 -> 向量化 -> FAISS检索 -> 返回最匹配接口
    """
    if index is None or faiss is None:
        print("[RAG检索跳过] FAISS索引不可用")
        return None
    
    query_emb = get_embedding(task) #输入的需求描述转成向量
    if query_emb is None:
        print("[RAG检索失败] 查询向量化失败")
        return None
    
    query_vector = np.array([query_emb], dtype=np.float32) #生成一个形状为 (1, 1024) 的二维矩阵
    #使用FAISS进行向量相似度搜索，返回最接近的接口索引和对应距离
    distances, indices = index.search(query_vector, min(top_k, len(docs))) #在所有接口向量里，找最相似的前k个
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(docs):
            doc = docs[idx] #根据索引，找到对应的接口信息（路径、名字等）
            doc["score"] = float(distances[0][i]) #给这个接口打一个“相似度分数”
            results.append(doc)
            print(f"[RAG检索] 匹配接口: {doc['name']} | 路径: {doc['path']} | 距离: {doc['score']:.4f}")
    
    return results[0] if results else None #最终返回最匹配的接口作为RAG上下文

def validate_case(testcases):
    """
    【校验模块】深度校验测试用例结构
    检查项: 空结果、字段完整性、input格式、expected格式
    返回: 校验结果 + 错误列表
    """
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



async def generate_with_feedback(task: str, base_prompt: str = "", max_retry: int = 2):
    """
    【自反馈生成函数】带重试机制的测试用例生成
    流程:
    1. 第一轮: 正常生成
    2. 校验失败 -> 第二轮: 基于base_prompt追加强化约束重新生成
    3. 返回最后一次有效结果

    为什么retry必须基于base_prompt而非重新拼接:
      retry时如果完全重新拼接Prompt，会丢失base_prompt中的:
      1. ROLE_DEFINITION - 角色定义
      2. RAG上下文 - 接口定义信息
      3. OUTPUT_FORMAT - 输出格式约束
      4. OUTPUT_EXAMPLE - 输出示例
      正确做法是: base_prompt（保留完整结构）+ retry约束（追加），
      通过build_retry_prompt统一构建，确保retry不丢失原始Prompt结构。

    输入:
      task: 用户需求（作为user message）
      base_prompt: 基础Prompt（来自build_prompt，作为system_message）
      max_retry: 最大重试次数
    """
    retry_count = 0
    last_testcases = None
    
    while retry_count < max_retry:
        print(f"\n{'='*50}")
        print(f"[第{retry_count + 1}轮生成] 开始生成测试用例...")
        print(f"{'='*50}")
        
        current_task = task
        current_base_prompt = base_prompt
        
        if retry_count == 1:
            print("[第2轮] 使用强化约束prompt（基于base_prompt追加约束）...")
            fail_reason = "上一轮生成结果未通过JSON校验"
            current_base_prompt = build_retry_prompt(base_prompt, fail_reason)
        
        testcases = await generate_testcases(current_task, current_base_prompt)
        #校验结构是否正常
        if testcases is None:
            print(f"[第{retry_count + 1}轮生成] ⚠️ 无法提取有效JSON，触发下一轮")
            retry_count += 1
            continue
        
        if isinstance(testcases, dict) and not testcases:
            print(f"[第{retry_count + 1}轮生成] ⚠️ 空JSON，触发下一轮")
            retry_count += 1
            continue
        #统计用例数量
        total = sum(len(testcases.get(k, [])) for k in ["normal", "abnormal", "boundary"])
        print(f"[第{retry_count + 1}轮生成] 提取到 {total} 个测试用例")
        #第二层校验：检查字段，格式
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
    return last_testcases #不完美也返回结果


def postprocess_testcases(testcases, max_cases=10):
    """
    【后处理模块】用例去重 + 数量控制
    去重策略: 基于method+path+input生成签名，过滤重复用例
    数量控制: 限制最大用例数，避免生成过多用例
    """
    def get_case_signature(case):
        """生成用例唯一签名，用于去重判断"""
        method = case.get("method", "")
        path = case.get("path", "")
        input_data = case.get("input", {})
        body = {k: v for k, v in input_data.items() if k not in ["headers", "query_params", "path_parameters"]}
        query = input_data.get("query_params", {})
        path_params = input_data.get("path_parameters", {})
        signature = {
            "method": method,
            "path": path,
            "body": json.dumps(body, sort_keys=True, ensure_ascii=False),
            "query": json.dumps(query, sort_keys=True, ensure_ascii=False),
            "path_params": json.dumps(path_params, sort_keys=True, ensure_ascii=False)
        }
        return json.dumps(signature, sort_keys=True, ensure_ascii=False)
    
    result = {"normal": [], "abnormal": [], "boundary": []}
    seen_signatures = set()
    total_count = 0
    
    for category in ["normal", "abnormal", "boundary"]:
        cases = testcases.get(category, [])
        for case in cases:
            if total_count >= max_cases:
                break
            signature = get_case_signature(case)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                result[category].append(case)
                total_count += 1
        if total_count >= max_cases:
            break
    
    original_total = sum(len(testcases.get(k, [])) for k in ["normal", "abnormal", "boundary"])
    dedup_count = original_total - len(seen_signatures)
    
    print(f"[后处理] 原始用例数: {original_total}, 去重: {dedup_count}, 最终: {total_count}")
    
    return result


def generate_json_testcases(testcases, base_url=None):
    """
    【JSON输出模块】将分类测试用例转换为标准化JSON格式
    
    为什么需要标准化JSON输出:
      1. 框架无关性：不绑定pytest，可转换为任意测试框架格式
      2. 便于审核：结构清晰，测试人员可直接阅读和修改
      3. 易于集成：可导入测试用例管理系统（如TestLink、Zephyr等）
      4. 支持扩展：新增字段不影响现有解析逻辑
    
    JSON字段说明:
      - title: 用例标题，简洁描述测试场景
      - method: HTTP方法（GET/POST/PUT/DELETE/PATCH）
      - path: 接口路径，支持路径参数占位符（如/pet/{petId}）
      - params: 请求参数，分为path/query/body三类
      - expected: 期望结果，包含status_code和body断言
      - case_type: 用例类型（正常场景/异常场景/边界场景）
    
    输入: 分类测试用例 {"normal": [...], "abnormal": [...], "boundary": [...]}
    输出: 标准化JSON列表
    """
    standard_cases = []
    
    case_type_mapping = {
        "normal": "正常场景",
        "abnormal": "异常场景",
        "boundary": "边界场景"
    }
    
    for category in ["normal", "abnormal", "boundary"]:
        cases = testcases.get(category, [])
        for case in cases:
            name = case.get("name", "未命名用例")
            method = case.get("method", "POST").upper()
            path = case.get("path", "")
            input_data = case.get("input", {})
            expected = case.get("expected", {})
            
            params = {
                "path": input_data.get("path_parameters", {}),
                "query": input_data.get("query_params", {}),
                "body": {k: v for k, v in input_data.items() 
                        if k not in ["headers", "query_params", "path_parameters"]}
            }
            
            if not params["path"]:
                del params["path"]
            if not params["query"]:
                del params["query"]
            if not params["body"]:
                del params["body"]
            
            expected_body = expected.get("body", {})
            if isinstance(expected_body, str):
                expected_body = {}
            
            standard_case = {
                "title": name,
                "method": method,
                "path": path,
                "params": params if params else {},
                "expected": {
                    "status_code": expected.get("status_code", 200),
                    "body": expected_body
                },
                "case_type": case_type_mapping.get(category, category)
            }
            
            standard_cases.append(standard_case)
    
    return standard_cases


def save_json_file(testcases, file_path=None):
    """
    【JSON输出模块】保存测试用例到JSON文件
    
    为什么单独封装保存逻辑:
      1. 统一编码处理：确保UTF-8编码，支持中文
      2. 统一格式化：indent=2保证可读性
      3. 便于扩展：后续可增加文件名校验、覆盖确认等逻辑
    
    输入: 测试用例列表, 目标文件路径
    输出: 实际保存的文件路径
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testcases.json')
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(testcases, f, ensure_ascii=False, indent=2)
    
    return file_path


def main():
    """
    【主入口函数】Streamlit Web应用主流程
    完整流程:
    1. 用户输入需求描述 + 可选Swagger文件
    2. RAG检索匹配接口（可选）
    3. Multi-Agent协作生成测试用例
    4. 校验 -> 后处理 -> 质量评估
    5. 输出标准化JSON测试用例文件
    """
    st.title("🤖 AI测试用例生成平台")
    st.markdown("**生成流程**: 输入需求 → RAG检索接口 → AI生成测试用例 → 输出JSON文件")
    st.divider()
    
    user_input = st.text_area(
        "📝 请输入需求描述",
        height=150,
        placeholder="例如：用户注册接口，需要用户名、密码、邮箱三个字段..."
    )
    
    DEFAULT_BASE_URL = "https://petstore.swagger.io/v2"
    
    user_base_url = st.text_input(
        "🔗 接口Base URL（可被Swagger覆盖）",
        value=DEFAULT_BASE_URL,
        help="接口基础URL，如提供Swagger文件将自动解析覆盖"
    )
    
    swagger_path = st.text_input(
        "📄 Swagger文件路径（可选）",
        value="",
        help="上传Swagger JSON文件路径，启用RAG接口匹配"
    )
    
    if st.button("🚀 生成测试用例", type="primary"):
        if not user_input:
            st.error("请输入需求描述！")
            return
        
        if not conf['deepseek']['api_key']:
            st.error("请先在config.ini中配置DeepSeek的API Key！")
            return
        
        final_base_url = None
        swagger_base_url = None
        swagger_docs = None
        api_info = None
        
        if swagger_path and os.path.exists(swagger_path):
            with st.spinner("🔍 RAG检索匹配接口（Hybrid Search + Rerank）..."):
                try:
                    pipeline = RAGPipeline(
                        base_url=conf['deepseek']['base_url'],
                        api_key=conf['deepseek']['api_key']
                    )
                    api_info = pipeline.run(swagger_path, user_input, top_k=3)
                    swagger_base_url = pipeline.base_url
                    swagger_docs = pipeline.docs
                    if swagger_base_url:
                        st.info(f"📌 Swagger解析 base_url: **{swagger_base_url}**")
                    if api_info:
                        st.info(f"📌 RAG匹配接口: **{api_info['name']}** ({api_info['method']} {api_info['path']})")
                    else:
                        st.warning("RAG未匹配到相关接口，使用原始需求生成")
                except Exception as e:
                    st.warning(f"RAG检索失败: {str(e)}，使用原始需求生成")
        
        if swagger_base_url and swagger_base_url.startswith("http"):
            final_base_url = swagger_base_url
        elif user_base_url and user_base_url.startswith("http"):
            final_base_url = user_base_url
        else:
            final_base_url = DEFAULT_BASE_URL
        
        st.info(f"🔗 最终使用的 base_url: **{final_base_url}**")
        
        with st.spinner("AI正在生成测试用例..."):
            base_prompt = build_prompt(api_info, user_input)
            writer_task = user_input
            
            try:
                testcases = asyncio.run(generate_with_feedback(writer_task, base_prompt))
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
        
        testcases = postprocess_testcases(testcases, max_cases=10)
        
        st.subheader("🔧 用例后处理")
        total_after = sum(len(testcases.get(k, [])) for k in ["normal", "abnormal", "boundary"])
        st.info(f"去重 + 数量控制后，最终用例数: **{total_after}**")
        
        quality_result = evaluate_testcases(testcases)
        
        st.subheader("📊 用例质量评估")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        col_q1.metric("JSON合法性", "✅ 合法" if quality_result["json_valid"] else "❌ 非法")
        col_q2.metric("结构完整率", f"{quality_result['structure_score']}")
        col_q3.metric("有效用例率", f"{quality_result['valid_case_rate']}")
        col_q4.metric("用例数", f"{quality_result['valid_cases']}/{quality_result['total_cases']}")
        
        coverage = quality_result["coverage"]
        st.markdown(f"**场景覆盖**: 正常场景 {'✅' if coverage['normal'] else '❌'} | 异常场景 {'✅' if coverage['abnormal'] else '❌'} | 边界场景 {'✅' if coverage['boundary'] else '❌'}")
        
        json_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data.json')
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(testcases, f, ensure_ascii=False, indent=2)
        st.info(f"✅ 测试用例已保存到: test_data.json")
        
        standard_testcases = generate_json_testcases(testcases, final_base_url)
        
        st.subheader("� 生成的标准化测试用例(JSON)")
        st.json(standard_testcases)
        
        json_output_path = save_json_file(standard_testcases)
        st.success(f"✅ 测试用例已保存到: **{json_output_path}**")
        
        st.subheader("📁 JSON文件说明")
        st.markdown("""
**输出文件**: `testcases.json`

**字段说明**:
| 字段 | 说明 |
|------|------|
| title | 用例标题，描述测试场景 |
| method | HTTP方法（GET/POST/PUT/DELETE/PATCH） |
| path | 接口路径 |
| params | 请求参数（path/query/body） |
| expected | 期望结果（status_code + body） |
| case_type | 用例类型（正常场景/异常场景/边界场景） |

**后续使用**:
- 可导入测试用例管理系统
- 可转换为pytest/JUnit等测试脚本
- 可用于API自动化测试框架
""")


if __name__ == "__main__":
    main()
