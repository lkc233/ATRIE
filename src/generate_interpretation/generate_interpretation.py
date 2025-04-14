import os
import json
import random
import configparser
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import sys

# Add custom module path
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../my_scripts'))

from chat import Chat, Chat_farui
from utils import load_docs, count_tokens, clean_ftgd

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.initialize_settings()
        self.load_legal_articles()
        self.initialize_chatbots()
        self.prompt_dict = {}
        
    def initialize_settings(self):
        """Initialize all settings from config"""
        self.input_folder = self.config['sec1']['input_folder']
        model_name = self.config['sec1']['model'].split('/')[-1]
        self.output_folder = os.path.join(
            self.config['sec1']['output_folder'], 
            str(self.config['sec1']['max_length']), 
            model_name, 
            self.config['sec1']['input_type']
        )
        self.input_type = self.config['sec1']['input_type']
        self.model = self.config['sec1']['model']
        self.sources = self.config['sec1']['sources'].split(', ')
        self.max_length = int(self.config['sec1']['max_length'])
        self.demo_path = self.config['sec1']['demo_path']
        self.case_ratio = float(self.config['sec1']['case_ratio'])
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_legal_articles(self):
        """Load legal articles from file"""
        with open('data/articles/法条.jsonl', 'r', encoding='utf-8') as f:
            articles = [json.loads(line) for line in f]
            self.articles_dict = {
                article['罪名']: article['法条编号'] + ' ' + article['法条内容'] 
                for article in articles
            }
            self.articles = articles
    
    def initialize_chatbots(self):
        """Initialize chatbot instances"""
        self.chatbots = [Chat(source=source) for source in self.sources]
    
    def generate_sampled_inputs(self, inputs):
        """Generate sampled inputs based on token count"""
        all_inputs = "\n\n".join(inputs)
        token_num = count_tokens(all_inputs, model=self.model)
        print(f"All Tokens: {token_num}")

        sampled_inputs = inputs
        while token_num > self.max_length:
            # 11/12:给prompt其他部分留有余地
            sample_size = int(11 / 12 * self.max_length * len(inputs)) // token_num
            sampled_inputs = random.sample(inputs, sample_size)
            all_inputs = "\n\n".join(sampled_inputs)
            token_num = count_tokens(all_inputs, model=self.model)

        return sampled_inputs, token_num
    
    def process_document(self, doc, crime, concept):
        """Process a single document for '概括案情和本院认为' type"""
        try:
            # 概括与模糊概念有关的案情
            prompt = f"""法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊概念进行具体化，并在裁判文书中进行分析。请你阅读裁判文书中的案情描述，并概括其中**与模糊概念有关**的案情。只需要输出概括后的案情即可。\n法条：{self.articles_dict[crime]}\n模糊概念：{concept}\n案情描述：{doc['ting_shen_guo_cheng']}\n"""
            chatbot = self.chatbots[int(doc['_id'])%len(self.chatbots)]
            response = chatbot.chat(prompt, model=self.model)

            # 处理法庭观点
            input_key = 'tsgc_ftgd'
            c_ftgd = clean_ftgd(doc['fa_ting_guan_dian']) if doc['fa_ting_guan_dian'] is not None else None

            # 更新文档的内容
            if c_ftgd is not None:
                doc[input_key] = f'案情描述：{response}\n法庭观点：{c_ftgd}\n'
            else:
                doc[input_key] = None
        except Exception as e:
            print(f"Error processing document: {e}")
            doc[input_key] = None
        return doc
    
    def process_documents_in_parallel(self, docs, crime, concept):
        """Process multiple documents in parallel"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self.process_document, doc, crime, concept) for doc in docs]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(docs), 
                             desc="Processing documents"):
                results.append(future.result())
        return results
    
    def prepare_inputs(self, crime, concept):
        """Prepare inputs based on input type"""
        doc_path = os.path.join(self.input_folder, f'{crime}_{concept}_train.jsonl')
        if not os.path.exists(doc_path):
            print(f'skip {crime}_{concept}')
            return None, None

        docs = load_docs(doc_path)
        docs = sorted(docs, key=lambda x: int(x['_id']))

        if self.input_type == '理由':
            reason_path = os.path.join(self.input_folder, f'reason/{crime}_{concept}_reason.jsonl')
            reasons = load_docs(reason_path)
            reasons = sorted(reasons, key=lambda x: int(x['_id']))
            assert len(docs) == len(reasons)

        input_key = self._get_input_key(docs)
        if input_key is None:
            return None, None

        if self.input_type == '理由':
            return self._prepare_reason_inputs(docs, reasons, crime, concept)
        elif self.input_type == '概括案情和本院认为':
            return self._prepare_summary_inputs(docs, crime, concept)
        else:
            inputs = [doc[input_key] if doc[input_key] is not None else "" for doc in docs]
            sampled_inputs, token_num = self.generate_sampled_inputs(inputs)
            sampled_inputs = random.sample(sampled_inputs, int(len(sampled_inputs)/self.case_ratio))
            return sampled_inputs, token_num
    
    def _get_input_key(self, docs):
        """Get the appropriate input key based on input type"""
        if self.input_type == '本院认为':
            input_key = 'fa_ting_guan_dian'
            for doc in docs:
                if doc[input_key] is not None:
                    doc[input_key] = clean_ftgd(doc['fa_ting_guan_dian'])
            return input_key
        elif self.input_type == '文书':
            input_key = 'wenshu'
            for i in range(len(docs)):
                docs[i][input_key] = None
                if 'ting_shen_guo_cheng' in docs[i] and 'fa_ting_guan_dian' in docs[i]:
                    if docs[i]['fa_ting_guan_dian'] is not None:
                        docs[i][input_key] = docs[i]['ting_shen_guo_cheng'] + clean_ftgd(docs[i]['fa_ting_guan_dian'])
            return input_key
        elif self.input_type in ['理由', '直接生成', '概括案情和本院认为']:
            return 'extracted_reason' if self.input_type == '理由' else self.input_type
        else:
            raise ValueError(f"Invalid input_type: {self.input_type}")
    
    def _prepare_reason_inputs(self, docs, reasons, crime, concept):
        """Prepare inputs for '理由' type"""
        for i in range(len(reasons)):
            assert docs[i]['_id'] == reasons[i]['_id']
            docs[i]['extracted_reason'] = reasons[i]['extracted_reason']
        
        inputs = [doc['extracted_reason'] if doc['extracted_reason'] is not None else "" for doc in docs]
        sampled_inputs, token_num = self.generate_sampled_inputs(inputs)
        sampled_inputs = random.sample(sampled_inputs, int(len(sampled_inputs)/self.case_ratio))
        return sampled_inputs, token_num
    
    def _prepare_summary_inputs(self, docs, crime, concept):
        """Prepare inputs for '概括案情和本院认为' type"""
        processed_docs_path = os.path.join(self.input_folder, f"{crime}_{concept}_train_processed.jsonl")
        if os.path.exists(processed_docs_path):
            with open(processed_docs_path, "r", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f]
        else:
            docs = self.process_documents_in_parallel(docs, crime, concept)
        
        input_key = 'tsgc_ftgd'
        inputs = [doc[input_key] if doc[input_key] is not None else "" for doc in docs]
        sampled_inputs, token_num = self.generate_sampled_inputs(inputs)
        
        # Save processed docs
        with open(processed_docs_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        return sampled_inputs, token_num
    
    def generate_prompt(self, crime, concept, sampled_inputs, token_num):
        """Generate prompt based on input type"""
        with open(self.demo_path, "r", encoding="utf-8") as f:
            demo_response = f.read()

        if self.input_type in ['本院认为', '文书', '理由']:
            input_json = {
                "法条": self.articles_dict[crime],
                "模糊概念": concept,
                "参考文本": sampled_inputs,
            }
            
            prompt1 = """法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊概念进行具体化，并在裁判文书中进行分析。请你阅读以下JSON数据，其中，"法条"是待分析的模糊概念所属的法条。"模糊概念"是你需要分析的具有模糊性的词语。"参考文本"是许多裁判文书中和模糊概念相关的文本。阅读完数据后，对法条中的模糊语词进行分析。"""
            prompt2 = json.dumps(input_json, ensure_ascii=False)
            prompt3 = f"请你对法条中的模糊语词的具体适用范围进行分析，请在输出中举例5个正例和5个反例，并详细地解释每个例子。案例说明部分请比参考分析中的更详细。<参考分析>\n{demo_response}\n</参考分析>\n"
            prompt = prompt1 + "\n\n" + prompt2 + "\n\n" + prompt3
        
        elif self.input_type == '直接生成':
            prompt1 = """法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊概念进行具体化，并在裁判文书中进行分析。请你阅读以下JSON数据，其中，"法条"是待分析的模糊概念所属的法条。"模糊概念"是你需要分析的具有模糊性的词语。"""
            input_json = {
                "法条": self.articles_dict[crime],
                "模糊概念": concept,
            }
            prompt2 = json.dumps(input_json, ensure_ascii=False)
            prompt3 = f"请你对法条中的模糊语词的具体适用范围进行分析，请在输出中举例5个正例和5个反例，并详细地解释每个例子。案例说明部分请比参考分析中的更详细。<参考分析>\n{demo_response}\n</参考分析>\n"
            prompt = prompt1 + "\n\n" + prompt2 + "\n\n" + prompt3
        
        elif self.input_type == '概括案情和本院认为':
            prompt1 = """法律语言的模糊性是其固有属性之一，而司法程序是对立法语言的一个明晰过程，法官会根据案件事实对法律条文中的模糊概念进行具体化，并在裁判文书中进行分析。请你阅读以下文本，其中，"法条"是待分析的模糊概念所属的法条。"模糊概念"是你需要分析的具有模糊性的词语。"案情描述"是裁判文书中与模糊概念有关的案件事实描述。"法庭观点"是法官在审理中对案件事实的分析，其中包含对模糊概念的认定。请阅读完文本后，对法条中的模糊概念进行分析。\n"""
            input2 = '\n'.join(sampled_inputs)
            prompt2 = f"""法条：{self.articles_dict[crime]}\n模糊概念：{concept}\n{input2}\n"""
            prompt3 = f"请你对法条中的模糊概念进行分析，请在输出中举例5个正例和5个反例，并详细地解释每个例子。案例说明部分请比参考分析中的更详细。<参考分析>\n{demo_response}\n</参考分析>\n"
            prompt = prompt1 + "\n\n" + prompt2 + "\n\n" + prompt3
        
        else:
            raise ValueError(f"Invalid input_type: {self.input_type}")

        return prompt
    
    def process_crime_concept(self, crime, concept):
        """Process a single crime-concept pair"""
        output_path = os.path.join(
            self.output_folder, f"{crime}_{concept}_response.txt")
        if os.path.exists(output_path):
            print(f"Skip {crime}_{concept}")
            return

        print(f"Processing {crime}_{concept} with {self.input_type}")
        
        sampled_inputs, token_num = self.prepare_inputs(crime, concept)
        if sampled_inputs is None:
            return

        prompt = self.generate_prompt(crime, concept, sampled_inputs, token_num)
        
        # Save prompt
        prompt_path = os.path.join(
            self.output_folder, f"{crime}_{concept}_{self.input_type}_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        self.prompt_dict[f"{crime}_{concept}"] = prompt
    
    def process_all_crimes(self):
        """Process all crime-concept pairs"""
        for article in self.articles:
            crime = article['罪名']
            concept = article['模糊概念']
            self.process_crime_concept(crime, concept)
        
        self.process_prompts_in_parallel()
    
    def process_prompts_in_parallel(self, max_workers=1):
        """Process all prompts in parallel"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.get_response, c_v)
                for c_v in self.prompt_dict.keys()
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    
    def get_response(self, c_v):
        """Get response for a single crime-concept pair"""
        crime, concept = c_v.split('_')[:2]
        output_path = os.path.join(
            self.output_folder, f"{crime}_{concept}_response.txt")
        if os.path.exists(output_path):
            return

        prompt = self.prompt_dict[c_v]
        chatbot = random.choice(self.chatbots)
        response = chatbot.chat(prompt, model=self.model)
        print(f"Response ok for {c_v}, input_type: {self.input_type}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Script to process and summarize documents.')
    parser.add_argument('-c', '--config', type=str, default='src/configs/summary/config.ini', help='Config file path')
    parser.add_argument('--input_folder', type=str, default= 'data/docs/v1/train', help='Path to the input folder')
    parser.add_argument('--output_folder', type=str, default= 'results/summary/', help='Path to the output folder')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Model to use')
    parser.add_argument('--sources', type=str, default='local_qwen', help='Source of the documents')
    parser.add_argument('--max_length', type=int, default=124000, help='Maximum length of the summary')
    parser.add_argument('--demo_path', type=str, default='few-shot-demo/example.txt', help='Path to demo file')
    parser.add_argument('--input_type', type=str, default='理由', help='doc type to generate summary')
    parser.add_argument('--case_ratio', type=float, default=1, help='case ratio to sample, 取倒数')
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update config with command line arguments"""
    if 'sec1' not in config:
        config['sec1'] = {}
    if args.input_folder:
        config['sec1']['input_folder'] = args.input_folder
    if args.output_folder:
        config['sec1']['output_folder'] = args.output_folder
    if args.input_type:
        config['sec1']['input_type'] = args.input_type
    if args.model:
        config['sec1']['model'] = args.model
    if args.sources:
        config['sec1']['sources'] = args.sources
    if args.max_length:
        config['sec1']['max_length'] = str(args.max_length)
    if args.demo_path:
        config['sec1']['demo_path'] = args.demo_path
    if args.case_ratio:
        config['sec1']['case_ratio'] = str(args.case_ratio)
    return config

def main():
    """Main function"""
    args = parse_arguments()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    config = update_config_with_args(config, args)
    
    processor = DocumentProcessor(config)
    processor.process_all_crimes()

if __name__ == "__main__":
    main()