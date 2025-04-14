# 自动评价（F1-score & Consistency Score）
import os
import shutil
import configparser
import argparse
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from sklearn.metrics import classification_report
import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../my_scripts'))
from utils import load_docs, extract_result, find_single_digit_numbers, all_modes
from chat import Chat


class DocumentEvaluator:
    def __init__(self, args):
        self.args = args
        self.config = self._load_config()
        self._setup_directories()
        self.chatbots = self._init_chatbots()
        self.labels = ['是', '否']
        self.total_price = 0
        self.articles = self._load_articles()
        
    def _load_config(self):
        config = configparser.ConfigParser()
        config.read(self.args.config)
        self._update_config_from_args(config)
        return config
        
    def _update_config_from_args(self, config):
        """根据命令行参数更新配置"""
        if 'sec1' not in config:
            config['sec1'] = {}
        arg_fields = ['input_type', 'input_dir', 'output_dir', 'model', 
                     'gt_dir', 'sources', 'reason_dir']
        for field in arg_fields:
            arg_value = getattr(self.args, field)
            if arg_value:
                config['sec1'][field] = arg_value
                
    def _setup_directories(self):
        """设置输出目录"""
        if os.path.exists(self.args.output_dir):
            shutil.rmtree(self.args.output_dir)
        os.makedirs(self.args.output_dir)
        self._save_config()
        
    def _save_config(self):
        """保存配置文件"""
        with open(os.path.join(self.args.output_dir, "config.ini"), "w", encoding="utf-8") as f:
            self.config.write(f)
            
    def _init_chatbots(self):
        sources = self.config['sec1']['sources'].split(', ')
        return [Chat(source=source) for source in sources]
        
    def _load_articles(self):
        """加载法条数据"""
        with open('data/articles/法条.jsonl', 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
            
    @staticmethod
    def most_frequent_element(lst):
        """获取列表中出现频率最高的元素"""
        if not lst:
            return None
        counter = Counter(lst)
        return counter.most_common(1)[0][0]
        
    def process_document(self, chatbots, doc, doc_gt, prompt, no_llm=False):
        """处理单个文档"""
        assert doc['id'] == doc_gt['id'], f"Doc id not match: {doc['id']} vs {doc_gt['id']}"
        chatbot = chatbots[int(doc['id'])%len(chatbots)]
        
        responses = doc.get('responses', [doc.get('response')])
        results = [extract_result(e) for e in responses]
        result = self.most_frequent_element(results)
        result_gt = extract_result(doc_gt['response'])
        
        # 处理无效结果
        if result is None:
            result = '否' if result_gt == '是' else '是'
            
        new_doc = doc.copy()
        new_doc.update({
            'result': result,
            'result_gt': result_gt,
            'eval_reason': None,
            'eval_score': 0
        })
        
        if no_llm:
            new_doc.update({
                'eval_input': "no_llm",
                'eval_reason': "no_llm",
                'eval_score': 0
            })
            return new_doc, result, result_gt, 0
            
        # 只对符合条件的进行处理
        if result in self.labels and result == result_gt:
            input_text = prompt + f"**模型生成的认定理由**：{doc['responses'][0]}\n**法庭观点**：{doc_gt['extracted_reason']}\n"
            eval_score = 0
            output_text = ""
            
            try:
                for _ in range(3):  # 尝试3次
                    output_text = chatbot.chat(input_text, self.args.model, with_usage=False)
                    matches = find_single_digit_numbers(output_text)
                    if len(matches) == 1:
                        eval_score = int(matches[0])
                        if 1 <= eval_score <= 10:
                            break
            except Exception as e:
                print(f"Error: {e}")
                
            new_doc.update({
                'eval_input': input_text,
                'eval_reason': output_text,
                'eval_score': eval_score
            })
            
        return new_doc, result, result_gt, 0
        
    def evaluate_crime_concept(self, crime, concept):
        """评估单个罪名和模糊概念组合"""
        print(f"Processing {crime}_{concept}")
        
        prompt = f"""请你参考法庭观点中对"{crime}"中的模糊概念"{concept}"的认定理由，对下面模型生成的认定理由的一致性进行1-10的打分。1分代表模型生成的认定理由和法庭观点中理由完全不一致，10分代表模型生成的认定理由和法庭观点中理由完全一致。请你先输出打分理由，然后以下列格式输出你的分数：[[n]]，其中n为你的分数。\n"""
        
        # 加载文档
        doc_path = os.path.join(f'{self.args.input_dir}/{self.args.input_type}', 
                              f'{crime}_{concept}.jsonl')
        doc_gt_path = os.path.join(self.args.gt_dir, '参考答案', 
                                 f'{crime}_{concept}.jsonl')
        
        docs = load_docs(doc_path)
        docs.sort(key=lambda x: x['id'])
        docs_gt = load_docs(doc_gt_path)
        docs_gt.sort(key=lambda x: x['id'])
        
        # 加载原因文档
        reason_docs = load_docs(os.path.join(self.args.reason_dir, f"{crime}_{concept}_reason.jsonl"))
        selected_ids = [doc['_id'] for doc in reason_docs]
        
        # 去重处理
        if len(selected_ids) != len(set(selected_ids)):
            unique_selected_ids = []
            unique_reason_docs = []
            for i in range(len(selected_ids)):
                if selected_ids[i] not in unique_selected_ids:
                    unique_selected_ids.append(selected_ids[i])
                    unique_reason_docs.append(reason_docs[i])
            selected_ids = unique_selected_ids
            reason_docs = unique_reason_docs
            
        reason_dict = {doc['_id']: doc for doc in reason_docs}
        
        # 过滤文档
        filtered_docs = []
        filtered_docs_gt = []
        for i in range(len(docs)):
            if docs[i]['id'] not in selected_ids:
                continue
                
            doc = docs[i]
            if not 'responses' in doc:
                doc['responses'] = [doc.get('response')]
                
            doc_gt = docs_gt[i]
            doc_gt['extracted_reason'] = reason_dict[doc_gt['id']]['extracted_reason']
            
            if not 'response' in doc_gt:
                doc_gt['response'] = doc_gt['responses'][0]
                
            result_gt = extract_result(doc_gt['response'])
            if result_gt not in self.labels:
                continue
                
            filtered_docs.append(doc)
            filtered_docs_gt.append(doc_gt)
            
        return self._process_documents(crime, concept, prompt, filtered_docs, filtered_docs_gt)
        
    def _process_documents(self, crime, concept, prompt, docs, docs_gt):
        """处理文档集合并评估"""
        results = []
        results_gt = []
        new_docs = docs
        scores = []
        
        max_workers = 64 if 'Qwen' in self.args.model else 32
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_document, self.chatbots, docs[i], docs_gt[i], prompt, False): i 
                for i in range(len(docs))
            }
            
            for future in tqdm(as_completed(future_to_doc), total=len(docs)):
                i = future_to_doc[future]
                new_doc, result, result_gt, price = future.result()
                new_docs[i] = new_doc
                results.append(result)
                results_gt.append(result_gt)
                self.total_price += price
                scores.append(new_doc['eval_score'])
                
        # 保存结果
        self._save_results(crime, concept, new_docs, results, results_gt, scores)
        return results, results_gt, scores
        
    def _save_results(self, crime, concept, docs, results, results_gt, scores):
        """保存评估结果"""
        # 保存JSONL文件
        with open(os.path.join(self.args.output_dir, f"{crime}_{concept}_{self.args.input_type}.jsonl"), "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                
        # 生成分类报告
        json_report = classification_report(results_gt, results, digits=4, output_dict=True, labels=self.labels)
        txt_report = classification_report(results_gt, results, digits=4, labels=self.labels)
        
        # 保存分类报告
        with open(os.path.join(self.args.output_dir, f"{crime}_{concept}_{self.args.input_type}.txt"), "w", encoding="utf-8") as f:
            f.write(txt_report)
            
        with open(os.path.join(self.args.output_dir, f"{crime}_{concept}_{self.args.input_type}.json"), "w", encoding="utf-8") as f:
            json.dump(json_report, f, ensure_ascii=False, indent=4)
            
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 追加到汇总文件
        path_parts = os.path.normpath(self.args.output_dir).split(os.sep)
        with open(os.path.join(self.args.output_dir, f"eval.txt"), "a", encoding="utf-8") as f:
            f.write(f"{crime}_{concept}:\n")
            f.write(f"{txt_report}\n")
            f.write(f"avg_score: {avg_score}\n")
            
    def run(self):
        """运行评估流程"""
        all_results = []
        all_results_gt = []
        all_scores = []
        
        for article in self.articles:
            crime = article['罪名']
            concept = article['模糊概念']
            
            results, results_gt, scores = self.evaluate_crime_concept(crime, concept)
            all_results.extend(results)
            all_results_gt.extend(results_gt)
            all_scores.extend(scores)
            
        # 保存总体结果
        self._save_final_results(all_results, all_results_gt, all_scores)
        
    def _save_final_results(self, all_results, all_results_gt, all_scores):
        """保存最终汇总结果"""
        json_report = classification_report(all_results_gt, all_results, digits=4, output_dict=True, labels=self.labels)
        txt_report = classification_report(all_results_gt, all_results, digits=4, labels=self.labels)
        
        path_parts = os.path.normpath(self.args.output_dir).split(os.sep)
        
        with open(os.path.join(self.args.output_dir, f"eval.txt"), "a", encoding="utf-8") as f:
            f.write(f"所有罪名:\n")
            f.write(f"{txt_report}\n")
            
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            none_zero_scores = [score for score in all_scores if score != 0]
            avg_score_wo_zero = sum(none_zero_scores) / len(none_zero_scores) if none_zero_scores else 0
            
            f.write(f"avg_score: {avg_score}\n")
            f.write(f"avg_score_wo_zero: {avg_score_wo_zero}\n")
            f.write(f"total_price: ${self.total_price}\n")
            
        with open(os.path.join(self.args.output_dir, f"eval.json"), "w", encoding="utf-8") as f:
            json.dump(json_report, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='src/configs/evaluate/config.ini', help='config file path')
    parser.add_argument('--input_type', type=str, default="理由", help='input type, overrides config file')
    parser.add_argument('--input_dir', type=str, default="results/pred/v1/Qwen2.5-72B-Instruct/Qwen2.5-14B-Instruct", help='input directory, overrides config file')
    parser.add_argument('--output_dir', type=str, default="results/pred/v1/Qwen2.5-72B-Instruct/Qwen2.5-14B-Instruct/理由/evaluation", help='output directory, overrides config file')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='model name, overrides config file')
    parser.add_argument('--sources', type=str, default="local_qwen, local_qwen2", help='source of chatbot, overrides config file')
    parser.add_argument('--gt_dir', type=str, default="results/pred/v1/Qwen2.5-72B-Instruct", help='ground truth directory, overrides config file')
    parser.add_argument('--reason_dir', type=str, default="data/docs/v1/test/reason", help='reason directory, overrides config file')
    
    args = parser.parse_args()
    evaluator = DocumentEvaluator(args)
    evaluator.run()
