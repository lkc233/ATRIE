import tiktoken
import json
import os
from http import HTTPStatus
import dashscope
import time
import re
from transformers import AutoTokenizer

def qwen_count_tokens(text,model,api_key=''):
    while True:
        response = dashscope.Tokenization.call(
            model=model,
            messages=[{'role': 'user', 'content': text}],
            api_key=api_key,
            )
        if response.status_code == HTTPStatus.OK:
            # print('Result is: %s' % response)
            break
        else:
            print('Failed request_id: %s, status_code: %s, code: %s, message:%s' %
                  (response.request_id, response.status_code, response.code,
                   response.message))
            if response.code == 'DataInspectionFailed':
                return 0
            time.sleep(1)
    return response.usage['input_tokens']

def qwen_local_count_tokens(text, model):
    tokenizer = AutoTokenizer.from_pretrained(model, )
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def count_tokens(text, model):
    if 'qwen' in model:
        return qwen_count_tokens(text, model)
    elif 'Qwen/' in model:
        return qwen_local_count_tokens(text, model)
    elif 'farui' in model:
        return qwen_local_count_tokens(text, 'Qwen/Qwen2.5-7B-Instruct')

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def load_docs(docs_path):
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return docs

# 获取当前文件夹路径
# current_path = os.path.abspath(os.path.dirname(__file__))
# with open(os.path.join(current_path, "articles.json"), "r", encoding="utf-8") as f:
#     articles = json.load(f)
# with open(os.path.join(current_path, "prompts_qwen_filter.json"), "r", encoding="utf-8") as f:
#     prompts_qwen_filter = json.load(f)
# with open(os.path.join(current_path, "prompts_pred.json"), "r", encoding="utf-8") as f:
#     prompts_pred = json.load(f)

# crime_filter = {
#     '为亲友非法牟利罪': ['为亲友非法牟利', '第一百六十六条'],
#     '危险驾驶罪': ['危险驾驶', '第一百三十三条'],
#     '盗窃罪': ['盗窃', '第二百六十四条'],
#     '强制猥亵、侮辱罪': ['强制猥亵', '侮辱', '第二百三十七条'],
#     '合同诈骗罪': ['合同诈骗', '第二百二十四条'],
#     '交通肇事罪': ['交通肇事', '第一百三十三条'],
#     '职务侵占罪': ['职务侵占', '第二百七十一条'],
#     '挪用资金罪': ['挪用资金', '第二百七十二条'],
# }

crime_filter = {
    '为亲友非法牟利罪': ['为亲友非法牟利', '第一百六十六条'],
    '危险驾驶罪': ['危险驾驶', '第一百三十三条'],
    '盗窃罪': ['盗窃', '第二百六十四条'],
    '强制猥亵、侮辱罪': ['强制猥亵', '侮辱', '第二百三十七条'],
    '合同诈骗罪': ['合同诈骗', '第二百二十四条'],
    '交通肇事罪': ['交通肇事', '第一百三十三条'],
    '职务侵占罪': ['职务侵占', '第二百七十一条'],
    '挪用资金罪': ['挪用资金', '第二百七十二条'],
}

vague_filter = {
    '为亲友非法牟利罪': {
        '亲友': ['亲友']
    },
    '危险驾驶罪': {
        '追逐竞驶、情节恶劣': ['追逐竞驶', '情节恶劣'],
        '机动车': ['机动车']
    },
    '盗窃罪': {
        '户': ['入户'],
        '凶器、携带': ['凶器', '携带'],
    },
    '强制猥亵、侮辱罪': {
        '公共场所当众': ['公共场所', '当众'],
    },
    '合同诈骗罪': {
        '合同': ['合同'],
    },
    '交通肇事罪': {
        '逃逸': ['逃逸'],
    },
    '职务侵占罪': {
        '职务': ['职务'],
    },
    '挪用资金罪': {
        '单位': ['单位'],
    },
}

wenshu_key_dict = {
    'shen_li_qing_kuang': '审理情况',
    'ting_shen_guo_cheng': '庭审过程',
    'fu_yan': '附言'
}


input_type2mode = {
    '只给案情': 'a', # Zero-shot
    '直接生成': 'b', # Direct Interpretation
    '司法解释': 'c', # Judicial Interpretation
    '本院认为': 'd', # ATRIE(Court View)
    '概括案情和本院认为': 'e', # ATRIE(Fact & Court View)
    '参考答案': 'gt', # ground-truth annotation(Sec 5.2 in our paper)
    'CoT': 'cot', # Chain of Thought
    '理由': 'f', # ATRIE
    'rag': 'g', 
    'random': 'r' # Random
}

all_modes = ['a', 'b', 'c', 'd', 'e', 'gt', 'cot', 'f', 'g', 'r']


def clean_ftgd(text):
    s = "判决如下"
    if s in text:
        # 找到“判决如下”出现的位置
        index = text.find(s)
        # 查找“判决如下”之前最后一个句号的位置
        last_period_index = text.rfind("。", 0, index)
        if last_period_index != -1:
            # 截取最后一个句号及其之前的部分
            text = text[:last_period_index + 1]
        else:
            # 如果没有找到句号，直接截取到“判决如下”前的位置
            text = text[:index]
    return text

def list_jsonl_files(directory='.'):
    """
    列出指定目录（默认为当前目录）下所有的jsonl文件

    Args:
        directory (str): 待遍历的目录，默认为当前目录 ('.')

    Returns:
        list: 返回jsonl文件名的列表
    """
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    return jsonl_files

# def find_jsonl_files_and_extract(directory):
#     # 使用正则表达式来匹配文件名格式 a_b.jsonl
#     pattern = re.compile(r'^(.*?)_(.*?)\.jsonl$')

#     result = []

#     # 遍历目录
#     for filename in os.listdir(directory):
#         if filename.endswith('.jsonl'):
#             match = pattern.match(filename)
#             if match:
#                 a, b = match.groups()
#                 result.append((a, b))
    
#     return result

def find_jsonl_files_and_extract(directory):
    result = []

    # 遍历目录
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            base_name = filename[:-6]  # 去掉后缀 '.jsonl'
            parts = base_name.split('_')
            parts = tuple(parts)
            result.append(parts)
    
    return result

def find_txt_files_and_extract(directory):
    result = []

    # 遍历目录
    for filename in os.listdir(directory):
        if filename.endswith('response.txt'):
            base_name = filename[:-4]  # 去掉后缀 '.jsonl'
            parts = base_name.split('_')
            parts = tuple(parts)
            result.append(parts)
    
    return result

def remove_json_tags(s):
    if s.startswith("```json") and s.endswith("```"):
        return s[7:-3].strip()
    return s

def get_unique_filepath(filepath):
    """
    接收完整的文件路径，检查文件名是否重复，如果重复则自动重命名文件。
    """
    directory, filename = os.path.split(filepath)  # 分离目录和文件名
    base, extension = os.path.splitext(filename)   # 分离文件名和扩展名
    counter = 1
    new_filename = filename
    
    # 检查文件是否存在
    while os.path.exists(os.path.join(directory, new_filename)):
        # 文件名后添加编号避免重复
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
        
    return os.path.join(directory, new_filename)

def extract_result(text):
    if text is None:
        return None
    if '[[是]]' in text:
        return '是'
    elif '[[否]]' in text:
        return '否'
    else:
        return None
    

def find_single_digit_numbers(text):
    # 正则表达式，匹配双括号内的一位整数
    pattern = r"\[\[(\d)\]\]"
    
    # 查找匹配
    matches = re.findall(pattern, text)
    
    # 返回结果
    return matches