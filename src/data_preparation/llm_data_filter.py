# LLM筛选与模糊概念相关的案件

import pickle
import json
import os
import sys
import argparse
from typing import Dict, List, Set, Any
import concurrent.futures
from tqdm.auto import tqdm
from collections import defaultdict
import time

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../my_scripts'))
from chat import Chat

class DocumentProcessor:
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", input_folder: str = 'data/docs/v0/'):
        self.model = model
        self.input_folder = input_folder
        self._setup_chat_instances()
        self._setup_processing_params()
        self.articles = self._load_articles()
        self.output_folder = self._init_output_folder()
        self.processed_ids = self._init_processed_ids()
        self.count = self._init_count_stats()

    def _setup_chat_instances(self) -> None:
        """Initialize chat instances based on model"""
        if self.model == "Qwen/Qwen2.5-7B-Instruct":
            self.chat_num = 4
        elif self.model == "Qwen/Qwen2.5-72B-Instruct":
            self.chat_num = 2
        else:
            raise ValueError("Invalid model name")

        self.chat1 = Chat(source='local_qwen')
        if self.chat_num == 4:
            self.chat2 = Chat(source='local_qwen2')
            self.chat3 = Chat(source='local_qwen3')
            self.chat4 = Chat(source='local_qwen4')
        elif self.chat_num == 2:
            self.chat2 = Chat(source='local_qwen2')

    def _setup_processing_params(self) -> None:
        """Set processing parameters based on chat instances"""
        if self.chat_num == 4:
            self.max_workers = 400
            # self.max_workers = 1
            self.times = 3
        else:
            self.max_workers = 80
            self.times = 1

    def _load_articles(self) -> List[Dict]:
        """Load articles from JSONL file"""
        article_path = 'data/articles/法条.jsonl'
        with open(article_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _init_output_folder(self) -> str:
        """Initialize output folder structure"""
        model_name = self.model.split('/')[-1]
        output_folder = os.path.join(self.input_folder, f'{model_name}_filtered')
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _init_processed_ids(self) -> Dict[str, Dict[str, Set[int]]]:
        """Initialize or load processed document IDs"""
        processed_ids_path = os.path.join(self.output_folder, 'processed_ids.pkl')
        if os.path.exists(processed_ids_path):
            return pickle.load(open(processed_ids_path, 'rb'))
        
        processed_ids = {}
        for article in self.articles:
            crime, concept = article['罪名'], article['模糊概念']
            if crime not in processed_ids:
                processed_ids[crime] = {concept: set()}
            else:
                assert concept not in processed_ids[crime]
                processed_ids[crime][concept] = set()
        
        pickle.dump(processed_ids, open(processed_ids_path, 'wb'))
        return processed_ids

    def _init_count_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Initialize statistics counter"""
        count = {}
        for crime, concepts in self.processed_ids.items():
            count[crime] = {}
            for concept in concepts:
                count[crime][concept] = {'pos': 0, 'neg': 0, 'total': 0}
        return count

    @staticmethod
    def get_pred_label(text: str) -> int:
        """Parse prediction label from response text"""
        if '[[是]]' in text:
            return 1
        elif '[[否]]' in text:
            return -1
        return 0

    def _get_chat_response(self, doc_id: int, prompt: str) -> str:
        """Get response from appropriate chat instance"""
        if self.chat_num == 4:
            if doc_id % 4 == 0:
                return self.chat1.chat(prompt, self.model, temperature=0.9)
            elif doc_id % 4 == 1:
                return self.chat2.chat(prompt, self.model, temperature=0.9)
            elif doc_id % 4 == 2:
                return self.chat3.chat(prompt, self.model, temperature=0.9)
            elif doc_id % 4 == 3:
                return self.chat4.chat(prompt, self.model, temperature=0.9)
        elif self.chat_num == 1:
            return self.chat1.chat(prompt, self.model, temperature=0.9)
        elif self.chat_num == 2:
            if doc_id % 2 == 0:
                return self.chat1.chat(prompt, self.model, temperature=0.9)
            elif doc_id % 2 == 1:
                return self.chat2.chat(prompt, self.model, temperature=0.9)
        else:
            raise ValueError("Invalid chat number")

    def process_document(self, doc: Dict, p_pos: str, p_neg: str) -> Dict:
        """Process a single document"""
        responses = {'pos': [], 'neg': []}
        
        for _ in range(self.times):
            response_pos = self._get_chat_response(doc['_id'], p_pos)
            response_neg = self._get_chat_response(doc['_id'], p_neg)
            responses['pos'].append(response_pos)
            responses['neg'].append(response_neg)
        
        doc['label'] = None
        for label, responses_ in responses.items():
            cnt = sum(self.get_pred_label(response) for response in responses_)
            if cnt > 0:
                doc['label'] = label
        
        doc['responses'] = responses
        return doc
    def _generate_prompts(self, crime: str, concept: str) -> tuple:
        """Generate positive and negative prompts for processing"""
        prompt_pos = f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊概念进行具体化并在裁判文书中的"法庭观点"部分进行分析。我们考虑模糊概念"{concept}"。我将给你一段法庭观点，请你判断法庭观点中：1. 是否有明确的句子表明案件中的情节被认定为罪名"{crime}"中的"{concept}"；\n2. 是否有具体的解释说明被告的行为属于"{concept}"的原因。\n先输出你的判断理由，然后严格按照以下格式输出你的最终判断。如果上面的两个条件均满足，输出"[[是]]"；如果其中的任一个条件不满足，输出"[[否]]"。\n\n【法庭观点】\n"""
        prompt_neg = f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊词进行具体化并在裁判文书中的"法庭观点"部分进行分析。我将给你一段法庭观点，请你判断法庭观点中是否有明确的句子表明案件中的情节不属于"{crime}"中的"{concept}"。先输出你的判断理由，然后严格按照以下格式输出你的最终判断。如果有明确的句子表明被告的行为不属于"{concept}"，输出"[[是]]"；如果没有，输出"[[否]]"。\n\n【法庭观点】\n"""
        return prompt_pos, prompt_neg

    def _update_stats(self, crime: str, concept: str, output_path: str) -> None:
        """Update statistics after processing documents"""
        with open(output_path, 'r', encoding='utf-8') as f:
            docs = [json.loads(line) for line in f]
        
        sorted_docs = sorted(docs, key=lambda x: int(x['_id']))
        self.count[crime][concept]['pos'] = len([doc for doc in sorted_docs if doc['label'] == 'pos'])
        self.count[crime][concept]['neg'] = len([doc for doc in sorted_docs if doc['label'] == 'neg'])
        self.count[crime][concept]['total'] = len(sorted_docs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in sorted_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    def process_article(self, article: Dict) -> None:
        """Process all documents for a given article"""
        crime, concept = article['罪名'], article['模糊概念']
        output_path = os.path.join(self.output_folder, f'{crime}_{concept}.jsonl')

        # Load documents to process
        input_path = os.path.join(self.input_folder, f'{crime}_{concept}.jsonl')
        with open(input_path, 'r', encoding='utf-8') as f:
            docs = [json.loads(line) for line in f]
            docs = [doc for doc in docs if doc['_id'] not in self.processed_ids[crime][concept]]

        # Generate prompts
        prompt_pos, prompt_neg = self._generate_prompts(crime, concept)

        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self.process_document,
                    doc,
                    prompt_pos + (doc['fa_ting_guan_dian'] or '法庭观点为空'),
                    prompt_neg + (doc['fa_ting_guan_dian'] or '法庭观点为空')
                )
                for doc in docs
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"Processing {crime}_{concept}"):
                doc = future.result()
                self.processed_ids[crime][concept].add(doc['_id'])
                pickle.dump(self.processed_ids, open(os.path.join(self.output_folder, 'processed_ids.pkl'), 'wb'))
                
                if doc['label'] is not None:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # Update statistics
        self._update_stats(crime, concept, output_path)

    def run(self) -> None:
        """Main processing pipeline"""
        start_time = time.time()  # 记录开始时间
        
        for article in self.articles:
            self.process_article(article)
        
        # Save final statistics
        with open(os.path.join(self.output_folder, 'count.json'), 'w', encoding='utf-8') as f:
            json.dump(self.count, f, ensure_ascii=False, indent=4)
        
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"Total running time: {elapsed_time:.2f} seconds")  # 打印运行时间

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加参数定义
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                      help='使用的模型名称，默认为 Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--input_folder', type=str, default='data/docs/v0/',
                      help='输入文件夹路径，默认为 data/docs/v0/')
    parser.add_argument('--max_workers', type=int, default=None,
                      help='最大工作线程数，不设置则使用类内部默认值')
    parser.add_argument('--times', type=int, default=None,
                      help='每个文档的处理次数，不设置则使用类内部默认值')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建处理器实例
    processor = DocumentProcessor(model=args.model, input_folder=args.input_folder)
    
    # 可选参数设置
    if args.max_workers is not None:
        processor.max_workers = args.max_workers
    if args.times is not None:
        processor.times = args.times
    
    # 运行处理流程
    processor.run()
