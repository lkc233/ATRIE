import json
from sklearn.model_selection import train_test_split
import os
import random
from typing import Dict, List, Any


class DataSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to a JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')

    def split_data(
        self,
        data: List[Dict[str, Any]],
        pos_max_len: int,
        neg_max_len: int
    ) -> tuple:
        """Split data into positive/negative and train/test sets."""
        pos_data = [item for item in data if item['label'] == 'pos']
        neg_data = [item for item in data if item['label'] == 'neg']

        # Sample data if exceeds max length
        pos_data = random.sample(pos_data, min(len(pos_data), pos_max_len))
        neg_data = random.sample(neg_data, min(len(neg_data), neg_max_len))

        # Split into train and test
        pos_train, pos_test = train_test_split(
            pos_data, test_size=0.5, random_state=self.seed)
        neg_train, neg_test = train_test_split(
            neg_data, test_size=0.5, random_state=self.seed)

        # Balance the datasets
        pos_train = random.sample(pos_train, min(len(pos_train), len(neg_train)))
        pos_test = random.sample(pos_test, min(len(pos_test), len(neg_test)))

        train_data = pos_train + neg_train
        test_data = pos_test + neg_test

        return train_data, test_data, len(pos_train), len(neg_train), len(pos_test), len(neg_test)

    def process_crime_concept(
        self,
        crime: str,
        concept: str,
        data_dir: str,
        output_dir: str,
        pos_max_len: int,
        neg_max_len: int,
        count: Dict[str, Any]
    ) -> None:
        """Process data for a single crime-concept pair."""
        file_path = os.path.join(data_dir, f'{crime}_{concept}.jsonl')
        data = self.load_jsonl(file_path)

        print(f'crime: {crime}, vague: {concept}')
        
        train_data, test_data, pos_train_len, neg_train_len, pos_test_len, neg_test_len = self.split_data(
            data, pos_max_len, neg_max_len)

        print(f'pos_train: {pos_train_len}, pos_test: {pos_test_len}')
        print(f'neg_train: {neg_train_len}, neg_test: {neg_test_len}')
        print(f'train: {len(train_data)}, test: {len(test_data)}\n')

        # Update count
        if crime not in count:
            count[crime] = {}
        count[crime][concept] = {'train': len(train_data), 'test': len(test_data)}

        # Save data
        train_path = os.path.join(output_dir, 'train', f'{crime}_{concept}_train.jsonl')
        test_path = os.path.join(output_dir, 'test', f'{crime}_{concept}_test.jsonl')
        self.save_jsonl(train_data, train_path)
        self.save_jsonl(test_data, test_path)

    def run(
        self,
        data_dir: str,
        output_dir: str,
        articles_path: str,
        pos_max_len: int = 300,
        neg_max_len: int = 300
    ) -> None:
        """Main method to run the data splitting process."""
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

        # Load articles
        articles = self.load_jsonl(articles_path)

        count = {}
        for article in articles:
            crime = article['罪名']
            concept = article['模糊概念']
            self.process_crime_concept(
                crime, concept, data_dir, output_dir, pos_max_len, neg_max_len, count)

        # Calculate total counts
        count['all'] = {'all': {'train': 0, 'test': 0}}
        for crime in count:
            if crime == 'all':
                continue
            for concept in count[crime]:
                count['all']['all']['train'] += count[crime][concept]['train']
                count['all']['all']['test'] += count[crime][concept]['test']

        # Save count statistics
        with open(os.path.join(output_dir, 'count.json'), 'w', encoding='utf-8') as f:
            json.dump(count, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    splitter = DataSplitter(seed=42)
    splitter.run(
        data_dir='data/docs/v0/Qwen2.5-7B-Instruct_filtered/Qwen2.5-72B-Instruct_filtered',
        output_dir='data/docs/v1',
        articles_path='data/articles/法条.jsonl'
    )
