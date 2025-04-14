# 判断未见过的案件中是否适用模糊概念
import os
import json
import configparser
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import random

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../my_scripts'))

from utils import load_docs, clean_ftgd, wenshu_key_dict, extract_result, input_type2mode, all_modes
from chat import Chat


class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.mode = input_type2mode[config['input_type']]
        self.articles_dict = {}
        self.sfjs_dict = {}
        self.chatbots = []
        self._load_resources()
        
    def _load_resources(self):
        """加载所有必要的资源"""
        self._load_articles()
        self._init_chatbots()
        
    def _load_articles(self):
        """加载法条数据"""
        with open('data/articles/法条.jsonl', 'r', encoding='utf-8') as f:
            articles = [json.loads(line) for line in f]
            
            for article in articles:
                crime = article['罪名']
                concept = article['模糊概念']
                self.articles_dict[crime] = article['法条编号'] + ' ' + article['法条内容']
                
                if '司法解释' in article:
                    self.sfjs_dict[f'{crime}_{concept}'] = article['司法解释']
                else:
                    self.sfjs_dict[f'{crime}_{concept}'] = ''
    
    def _init_chatbots(self):
        sources = self.config['sources'].split(', ')
        self.chatbots = [Chat(source=source) for source in sources]
    
    def _get_reference_doc(self, crime, concept):
        """获取参考文档内容"""
        if self.mode in ['a', 'gt', 'c', 'cot', 'r']:
            if self.mode == 'c':
                return self.sfjs_dict.get(f'{crime}_{concept}', '')
            return ''
        
        ref_path = os.path.join(self.config['ref_dir'], self.config['input_type'], f'{crime}_{concept}_response.txt')
        if not os.path.exists(ref_path):
            print(f"No reference found for {crime}_{concept}")
            print(ref_path)
            return None
        
        with open(ref_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _generate_prompt(self, crime, concept, ref_doc, doc):
        doc['clean_fa_ting_guan_dian'] = clean_ftgd(doc['fa_ting_guan_dian'])
        
        wenshu_keys = ['shen_li_qing_kuang', 'ting_shen_guo_cheng', 'fu_yan']
        fact = '**审理情况、庭审过程、附言**：\n' + \
                 '\n\n'.join(['- ' + wenshu_key_dict[key]+': \n'+doc[key] 
                            if doc[key] is not None else '' for key in wenshu_keys])

        prompts = {
            'w': f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的“法庭观点”部分进行分析。在法条“{self.articles_dict[crime]}”中，模糊概念是“{concept}”。请你阅读下面对模糊概念的解释，根据裁判文书中的审理情况、庭审过程和附言，判断案件中的情况是否符合模糊概念“{concept}”的定义。先提供理由，然后严格按照以下格式输出你的最终判断：“[[是]]”如果符合模糊概念“{concept}”的定义，“[[否]]”如果不符合。\n\n<模糊概念的解释>\n{ref_doc}\n</模糊概念的解释>\n\n""", 
            'wo': f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的“法庭观点”部分进行分析。在法条“{self.articles_dict[crime]}”中，模糊概念是“{concept}”。请你根据裁判文书中的审理情况、庭审过程和附言，判断案件中的情况是否符合模糊概念“{concept}”的定义。先提供理由，然后严格按照以下格式输出你的最终判断：“[[是]]”如果符合模糊概念“{concept}”的定义，“[[否]]”如果不符合。\n\n""",
            'gt': f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的“法庭观点”部分进行分析。在法条“{self.articles_dict[crime]}”中，模糊概念是“{concept}”。请你根据裁判文书中的法庭观点，判断法官认为案件中的情况是否符合模糊概念“{concept}”的定义。先提供理由，然后严格按照以下格式输出你的最终判断：“[[是]]”如果符合模糊概念“{concept}”的定义，“[[否]]”如果不符合。\n\n""",
            'c': f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的“法庭观点”部分进行分析。在法条“{self.articles_dict[crime]}”中，模糊概念是“{concept}”。请你阅读下面和模糊概念相关的司法解释，根据裁判文书中的审理情况、庭审过程和附言，判断案件中的情况是否符合模糊概念“{concept}”的定义。先提供理由，然后严格按照以下格式输出你的最终判断：“[[是]]”如果符合模糊概念“{concept}”的定义，“[[否]]”如果不符合。\n\n<模糊概念的解释>\n{ref_doc}\n</模糊概念的解释>\n\n""",
            'cot': f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的“法庭观点”部分进行分析。在法条“{self.articles_dict[crime]}”中，模糊概念是“{concept}”。请你根据裁判文书中的审理情况、庭审过程和附言，判断案件中的情况是否符合模糊概念“{concept}”的定义。先提供理由，然后严格按照以下格式输出你的最终判断：“[[是]]”如果符合模糊概念“{concept}”的定义，“[[否]]”如果不符合。请一步一步的分析，先解释模糊概念，再将其具体应用到案件中。\n\n"""
        }
        
        inputs = {
            'a': prompts['wo'] + fact,
            'b': prompts['w'] + fact,
            'd': prompts['w'] + fact,
            'e': prompts['w'] + fact,
            'f': prompts['w'] + fact,
            'gt': prompts['gt'] + doc['clean_fa_ting_guan_dian'],
            'c': prompts['c'] + fact,
            'cot': prompts['cot'] + fact
        }
        
        return inputs.get(self.mode, prompts['wo'] + fact)
    
    def process_document(self, doc, crime, concept):
        """处理单个文档"""
        if doc['fa_ting_guan_dian'] is None:
            return None
        
        if self.mode == 'r':
            label = random.choice(['是', '否'])
            response = f"[[{label}]]"
            return {
                'id': doc['_id'],
                'input': '',
                'responses': [response],
            }
        
        ref_doc = self._get_reference_doc(crime, concept)
        if ref_doc is None:  # 仅当_get_reference_doc返回None时
            return None
        
        input_ = self._generate_prompt(crime, concept, ref_doc, doc)
        chatbot = self.chatbots[int(doc['_id'])%len(self.chatbots)]
        
        if self.mode == 'gt':
            responses = []
            answers = []
            for _ in range(self.config['response_num']):
                response = chatbot.chat(input_, model=self.config['model'], temperature=0.9)
                responses.append(response)
                answers.append(extract_result(response))
            
            num_yes = answers.count('是')
            num_no = answers.count('否')
            
            if num_yes > num_no:
                response = responses[answers.index('是')]
            else:
                response = responses[answers.index('否')]
            
            if num_yes == 0 and num_no == 0:
                print(f"No answer found for {crime}_{concept} doc id: {doc['_id']}")
                response += '\n\n' + '没有找到答案。[[否]]'
            responses = [response]
        else:
            responses = [
                chatbot.chat(input_, model=self.config['model'], temperature=0.3)
                for _ in range(self.config['response_num'])
            ]
        
        return {
            'id': doc['_id'],
            'input': input_,
            'responses': responses,
        }


class PredictionPipeline:
    def __init__(self, config):
        self.config = config
        self.processor = DocumentProcessor(config)
        self._setup_directories()
    
    def _setup_directories(self):
        """设置输入输出目录"""
        if self.processor.mode == 'gt':
            self.output_dir = os.path.join(self.config['output_dir'], self.config['input_type'])
        else:
            model_name = self.config['model'].split('/')[-1]
            self.output_dir = os.path.join(self.config['output_dir'], 
                                         model_name, 
                                         self.config['input_type'])
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _save_config(self):
        """保存配置文件"""
        config = configparser.ConfigParser()
        config['sec1'] = self.config
        with open(os.path.join(self.output_dir, 'config.ini'), 'w') as f:
            config.write(f)
    
    def _process_crime_concept(self, crime, concept):
        """处理单个罪名和模糊概念组合"""
        print(f"Processing {crime}_{concept}")
        doc_path = os.path.join(self.config['input_dir'], f'{crime}_{concept}_test.jsonl')
        docs = load_docs(doc_path)[:]
        print(f'Docs: {len(docs)}')
        
        output_path = os.path.join(self.output_dir, f"{crime}_{concept}.jsonl")
        count_path = os.path.join(self.output_dir, f"{crime}_{concept}_count.json")
        
        # 检查进度
        num = self._check_progress(count_path, docs)
        if num == len(docs):
            print(f"Skip {crime}_{concept}")
            return
        
        # 处理文档
        self._process_documents(docs, crime, concept, output_path, count_path)
    
    def _check_progress(self, count_path, docs):
        """检查处理进度"""
        if not os.path.exists(count_path):
            with open(count_path, 'w', encoding='utf-8') as f:
                json.dump({'num': 0}, f, ensure_ascii=False)
            return 0
        
        with open(count_path, 'r', encoding='utf-8') as f:
            return json.load(f)['num']
    
    def _process_documents(self, docs, crime, concept, output_path, count_path):
        """处理文档并保存结果"""
        if os.path.exists(output_path):
            os.remove(output_path)
        
        num = 0
        with ThreadPoolExecutor(max_workers=200) as executor:
            futures = [
                executor.submit(self.processor.process_document, doc, crime, concept)
                for doc in docs[:]
            ]
            
            with open(output_path, "a", encoding="utf-8") as f:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        num += 1
                        with open(count_path, 'w', encoding='utf-8') as f2:
                            json.dump({'num': num}, f2, ensure_ascii=False)
    
    def run(self):
        """运行整个处理流程"""
        self._save_config()
        
        with open('data/articles/法条.jsonl', 'r', encoding='utf-8') as f:
            articles = [json.loads(line) for line in f]
            
            for article in articles:
                crime = article['罪名']
                concept = article['模糊概念']
                self._process_crime_concept(crime, concept)

def parse_args():
    """解析命令行参数"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                       default='src/configs/pred/config.ini', 
                       help='config file path')
    parser.add_argument('--input_type', type=str, default='参考答案', help='Type of input data')
    parser.add_argument('--input_dir', type=str, default='data/docs/v1/test', help='Directory for input data')
    parser.add_argument('--output_dir', type=str, default='results/pred/v1/Qwen2.5-72B-Instruct', help='Directory for output data')
    parser.add_argument('--ref_dir', type=str, default='results/summary/124000/Qwen2.5-72B-Instruct', help='Directory for reference data')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Model to use')
    parser.add_argument('--sources', type=str, default='local_qwen, local_qwen2', help='Source of the data')
    parser.add_argument('--response_num', type=int, default=1, help='Number of responses to generate')
    
    args = parser.parse_args()
    
    # 加载配置文件并合并命令行参数
    config = configparser.ConfigParser()
    config.read(args.config)
    
    config_dict = {
        'input_type': args.input_type if args.input_type else config['sec1']['input_type'],
        'input_dir': args.input_dir if args.input_dir else config['sec1']['input_dir'],
        'output_dir': args.output_dir if args.output_dir else config['sec1']['output_dir'],
        'ref_dir': args.ref_dir if args.ref_dir else config['sec1']['ref_dir'],
        'model': args.model if args.model else config['sec1']['model'],
        'sources': args.sources or config['sec1']['sources'],
        'response_num': args.response_num if args.response_num else int(config['sec1']['response_num'])
    }
    
    return config_dict


def main():
    config = parse_args()
    pipeline = PredictionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()