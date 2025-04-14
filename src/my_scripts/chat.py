from openai import OpenAI
import anthropic
import yaml
import os
import time
import google.generativeai as genai
import dashscope    

class ChatConfigManager:
    """处理配置文件的加载和代理设置"""
    SUPPORTED_SOURCES = ['openai', 'qwen', 'local_qwen', 'local_qwen2', 
                        'local_qwen3', 'local_qwen4', 'local_glm', 'farui']
    PROXY_REQUIRED_SOURCES = ['openai', 'gemini']
    
    @classmethod
    def load_config(cls, config_path=None):
        """加载YAML配置文件"""
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '../configs/llm/config.yaml')
            
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    
    @classmethod
    def setup_proxy(cls, config, requires_proxy):
        """设置或取消HTTP代理"""
        if requires_proxy:
            os.environ['http_proxy'] = config['proxy']['http']
            os.environ['https_proxy'] = config['proxy']['https']
        else:
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)

class ChatClientFactory:
    """创建不同API的客户端实例"""
    
    @staticmethod
    def create_client(source, config):
        """根据来源创建对应的API客户端"""
        api_config = config['api_key'][source]
        
        if source in ChatConfigManager.SUPPORTED_SOURCES:
            return OpenAI(
                api_key=api_config['key'],
                base_url=api_config['base_url']
            )
        elif source == 'claude':
            return anthropic.Anthropic(
                api_key=api_config['key'],
                base_url=api_config['base_url']
            )
        elif source == 'gemini':
            genai.configure(api_key=api_config['key'])
            return None
        return None

class Chat:
    """主聊天类，处理与不同AI模型的交互"""
    MAX_RETRY_ATTEMPTS = 10
    DEFAULT_SYSTEM_PROMPT = "你是人工智能法律助手，能够回答与中国法律相关的问题。"
    
    def __init__(self, config=None, source='openai'):
        self.source = source
        self.config = ChatConfigManager.load_config(config)
        
        requires_proxy = source in ChatConfigManager.PROXY_REQUIRED_SOURCES
        ChatConfigManager.setup_proxy(self.config, requires_proxy)
        
        self.client = ChatClientFactory.create_client(source, self.config)

    def _handle_openai_response(self, model, messages, temperature, with_usage):
        """处理OpenAI兼容API的响应"""
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=temperature,
            timeout=60000000
        )
        if with_usage:
            return completion.usage, completion.choices[0].message.content
        return completion.choices[0].message.content

    def _handle_claude_response(self, model, messages, temperature):
        """处理Claude API的响应"""
        message = self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return message.content[0].text

    def _handle_gemini_response(self, model, system_prompt, user_prompt):
        """处理Gemini API的响应"""
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(system_prompt + user_prompt)
        return response.text

    def chat(self, prompt, model=None, system_prompt=None, temperature=0.0, with_usage=False):
        """主聊天方法"""
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if 'o1' in model:
            messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                if self.source in ChatConfigManager.SUPPORTED_SOURCES:
                    return self._handle_openai_response(model, messages, temperature, with_usage)
                elif self.source == 'claude':
                    return self._handle_claude_response(model, messages, temperature)
                elif self.source == 'gemini':
                    return self._handle_gemini_response(model, system_prompt, prompt)
            except Exception as error:
                if error.args[0] != 'KeyboardInterrupt':
                    print(f'Chat Error: {error}')
                    time.sleep(5)
        return None

class Chat_farui:
    """法睿AI专用聊天类"""
    SUPPORTED_MODELS = ['farui-plus']
    
    def __init__(self, api_key=''):
        self.api_key = api_key

    def chat(self, prompt, model='farui-plus', system_prompt=None, 
             temperature=0.0, with_usage=False):
        """与法睿AI交互的主方法"""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid model name. Supported models: {self.SUPPORTED_MODELS}')
            
        system_prompt = system_prompt or Chat.DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(Chat.MAX_RETRY_ATTEMPTS):
            try:
                response = dashscope.Generation.call(
                    model,
                    messages=messages,
                    result_format='message',
                    api_key=self.api_key,
                    temperature=temperature,
                )
                if with_usage:
                    return response.usage, response.output.choices[0].message.content
                return response.output.choices[0].message.content
            except Exception as error:
                if error.args[0] != 'KeyboardInterrupt':
                    print(f'Chat Error: {error}')
                    time.sleep(5)
        return None

if __name__ == '__main__':
    TEST_PROMPT = "你好，法定结婚年龄是多少？"
    CONFIG_PATH = 'configs/llm/config.yaml'
    
    for _ in range(1):
        print('gpt-4o-mini')
        chat_instance = Chat()
        response = chat_instance.chat(TEST_PROMPT, model='gpt-4o-mini')
        print(response)
