# 抽取案件中解释模糊概念是否适用的文段

import os
import json
import configparser
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../my_scripts'))
from utils import load_docs, clean_ftgd
from chat import Chat


class ReasonExtractor:
    def __init__(self):
        self.args = self._parse_args()
        self.config = self._load_config()
        self.articles, self.articles_dict = self._load_articles()
        self.chatbots = self._init_chatbots()
        self.progress = {}  # 改为字典，按crime_concept存储已处理的文档ID
        self.progress_file = os.path.join(self.args.output_dir, 'progress.json')  # 进度文件路径
        
    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, 
                          default='src/configs/extract_reason/config.ini',
                          help='Config file path')
        parser.add_argument('--input_dir', type=str, 
                          default='data/docs/v1/test',
                          help='Directory for input data')
        parser.add_argument('--output_dir', type=str, 
                          default='data/docs/v1/test/reason',
                          help='Directory for output data')
        parser.add_argument('--model', type=str, 
                          default='Qwen/Qwen2.5-72B-Instruct', 
                          help='Model to use')
        parser.add_argument('--sources', type=list, 
                          default=['local_qwen', 'local_qwen2'], 
                          help='Sources of the data')
        parser.add_argument('--resume', action='store_true',
                          help='Resume from last progress')
        return parser.parse_args()

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read(self.args.config)
        if 'sec1' not in config:
            config['sec1'] = {}
        # Update config with command line args
        config['sec1']['input_dir'] = self.args.input_dir or config['sec1']['input_dir']
        config['sec1']['output_dir'] = self.args.output_dir or config['sec1']['output_dir']
        config['sec1']['model'] = self.args.model or config['sec1']['model']
        config['sec1']['sources'] = ', '.join(self.args.sources) if self.args.sources else config['sec1']['sources']
        
        return config

    def _load_articles(self):
        articles_path = 'data/articles/法条.jsonl'
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = [json.loads(line) for line in f]
            return articles, {
                article['罪名']: article['法条编号'] + ' ' + article['法条内容'] 
                for article in articles
            }

    def _init_chatbots(self):
        sources = self.args.sources if self.args.sources else self.config['sec1']['sources'].split(', ')
        return [Chat(source=source) for source in sources]

    def _create_output_dir(self):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
            
        # Save config
        with open(os.path.join(self.args.output_dir, 'config.ini'), 'w') as f:
            self.config.write(f)

    def _load_progress(self):
        """加载处理进度"""
        if not self.args.resume or not os.path.exists(self.progress_file):
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def _save_progress(self):
        """保存处理进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def _process_single_document(self, doc, crime, concept):
        crime_concept = f"{crime}_{concept}"
        # 如果文档已处理过，则跳过
        if crime_concept in self.progress and doc['_id'] in self.progress[crime_concept]:
            return None
            
        if doc['fa_ting_guan_dian'] is None:
            # 添加到对应crime_concept的处理记录中
            if crime_concept not in self.progress:
                self.progress[crime_concept] = []
            self.progress[crime_concept].append(doc['_id'])
            return None

        doc['clean_fa_ting_guan_dian'] = clean_ftgd(doc['fa_ting_guan_dian'])

        prompt_gt = f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的"法庭观点"部分进行分析。在法条"{self.articles_dict[crime]}"中，模糊概念是"{concept}"。请你阅读裁判文书中的法庭观点，提取出法官对模糊概念的认定理由。比如，如果模糊概念是"户"，你需要提取出法官认为案件中满足或不满足"户"的理由是什么。理由包括对案件事实经过的分析和最后的结论。你的输出应该只有提取出的理由，并且为裁判文书中的原文\n\n【法庭观点】\n"""

        input_gt = prompt_gt + doc['clean_fa_ting_guan_dian']
        chatbot = self.chatbots[doc['_id']%len(self.chatbots)]
        response = chatbot.chat(input_gt, model=self.args.model, temperature=0.9)
        
        # 添加到对应crime_concept的处理记录中
        if crime_concept not in self.progress:
            self.progress[crime_concept] = []
        self.progress[crime_concept].append(doc['_id'])
        
        return {
            '_id': doc['_id'],
            'clean_fa_ting_guan_dian': doc['clean_fa_ting_guan_dian'],
            'extracted_reason': response,
            'label': doc['label']
        }

    def process_crime_concept(self, article):
        crime = article['罪名']
        concept = article['模糊概念']
        crime_concept = f"{crime}_{concept}"
        print(f"Processing {crime_concept}")
        
        data_type = os.path.basename(self.args.input_dir)
        doc_path = os.path.join(self.args.input_dir, f'{crime_concept}_{data_type}.jsonl')
        docs = load_docs(doc_path)[:]
        print(f'Total docs: {len(docs)}')
        
        # 过滤掉已处理的文档
        if self.args.resume and crime_concept in self.progress:
            processed_ids = set(self.progress[crime_concept])
            docs = [doc for doc in docs if doc['_id'] not in processed_ids]
            print(f'Remaining docs to process: {len(docs)}')

        output_path = os.path.join(self.args.output_dir, f"{crime_concept}_reason.jsonl")
        
        # 如果是继续模式，不需要删除已有输出文件
        if not self.args.resume and os.path.exists(output_path):
            os.remove(output_path)

        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = [
                executor.submit(self._process_single_document, doc, crime, concept)
                for doc in docs
            ]
            
            with open(output_path, "a", encoding="utf-8") as f:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 保存进度
        self._save_progress()

    def run(self):
        self._create_output_dir()
        
        # 加载已有进度
        if self.args.resume:
            self.progress = self._load_progress()
            print(f"Resuming from previous progress")
        
        for article in self.articles:
            self.process_crime_concept(article)


def main():
    extractor = ReasonExtractor()
    extractor.run()


if __name__ == "__main__":
    main()
