"""
从裁判文书数据库中使用字符串匹配初筛和模糊概念有关的裁判文书
对应版本data/docs/v0
"""

import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import pymongo


class JudgmentDocumentFilter:
    """裁判文书过滤器类"""

    def __init__(self, config):
        """初始化配置"""
        self.config = config
        self.document_keys = [
            'biao_ti', 
            'fa_ting_guan_dian',
            'shen_li_qing_kuang', 
            'ting_shen_guo_cheng', 
            'pan_jue_jie_guo'
        ]
        self.crime_counts = defaultdict(lambda: defaultdict(int))
        self.total_crime_counts = defaultdict(int)
        self.concept_ratio_counts = defaultdict(lambda: defaultdict(float))
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_legal_articles(self):
        """加载法条数据"""
        with open(self.config.article_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def get_mongo_query(self):
        """获取MongoDB查询条件"""
        return {
            'an_jian_lei_xing': '刑事案件', 
            'biao_ti': {'$regex': '判决'},
        }

    def connect_to_mongodb(self):
        """连接MongoDB数据库"""
        client = pymongo.MongoClient(self.config.mongo_host, self.config.mongo_port)
        return client[self.config.db_name][self.config.collection_name]

    def process_document(self, document, articles):
        """处理单个裁判文书文档"""
        if not document.get('fa_ting_guan_dian'):
            return
            
        document_texts = [
            document[key] for key in self.document_keys 
            if key in document and document[key] is not None
        ]
        
        for article in articles:
            crime = article['罪名']
            concept = article['模糊概念']
            article_number = article['法条编号']
            
            # 检查是否匹配罪名或法条编号
            crime_found = any(
                val in doc 
                for doc in document_texts 
                for val in [crime, article_number]
            )
            
            if crime_found:
                self.total_crime_counts[crime] += 1
                
                if concept in document['fa_ting_guan_dian']:
                    # 如果达到最大数量则停止
                    if self.crime_counts[crime][concept] >= self.config.max_documents_per_concept:
                        continue

                    self.crime_counts[crime][concept] += 1
                    self._save_matching_document(crime, concept, document)

    def _save_matching_document(self, crime, concept, document):
        """保存匹配的文档"""
        output_file = os.path.join(
            self.config.output_dir, 
            f'{crime}_{concept}.jsonl'
        )
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(document, ensure_ascii=False) + '\n')

    def calculate_ratios(self):
        """计算概念出现比例"""
        for crime in self.crime_counts:
            for concept in self.crime_counts[crime]:
                if self.total_crime_counts[crime] > 0:
                    self.concept_ratio_counts[crime][concept] = (
                        self.crime_counts[crime][concept] / self.total_crime_counts[crime]
                    )

    def save_results(self):
        """保存结果统计"""
        results = [
            ('count.json', self.crime_counts),
            ('all_count.json', self.total_crime_counts),
            ('ratio_count.json', self.concept_ratio_counts),
        ]
        
        for filename, data in results:
            output_path = os.path.join(self.config.output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def run(self):
        """运行主流程"""
        articles = self.load_legal_articles()
        collection = self.connect_to_mongodb()
        query = self.get_mongo_query()
        
        for document in tqdm(collection.find(query)):
            self.process_document(document, articles)
            
        self.calculate_ratios()
        self.save_results()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='裁判文书模糊概念筛选器')
    
    parser.add_argument('--article_path', type=str, 
                       default='data/articles/法条.jsonl',
                       help='法条数据文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='data/docs/v0',
                       help='输出目录路径')
    parser.add_argument('--mongo_host', type=str,
                       default='localhost',
                       help='MongoDB主机地址')
    parser.add_argument('--mongo_port', type=int,
                       default=27017,
                       help='MongoDB端口号')
    parser.add_argument('--db_name', type=str,
                       default='cpws',
                       help='数据库名称')
    parser.add_argument('--collection_name', type=str,
                       default='main',
                       help='集合名称')
    parser.add_argument('--max_documents_per_concept', type=int,
                       default=100000,
                       help='每个概念最大文档数量')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    filter = JudgmentDocumentFilter(args)
    filter.run()
