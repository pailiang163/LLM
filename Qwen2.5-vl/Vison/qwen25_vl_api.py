#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL API调用示例
支持文本对话、图像理解、视频处理等功能
"""

import os
import base64
from openai import OpenAI
import dashscope
from typing import Optional, List, Dict, Any

# 设置API密钥和基础URL
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("请设置环境变量 DASHSCOPE_API_KEY")
    exit(1)

# OpenAI兼容方式的客户端
openai_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# DashScope原生客户端
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


class Qwen25VLClient:
    """Qwen2.5-VL API客户端封装"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def encode_image(self, image_path: str) -> str:
        """将本地图像编码为Base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def text_chat(self, 
                  message: str, 
                  model: str = "qwen-vl-max-latest",
                  system_prompt: str = "You are a helpful assistant.") -> str:
        """纯文本对话"""
        try:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"文本对话出错: {str(e)}"
    
    def image_understanding(self, 
                          image_url: str, 
                          text_prompt: str,
                          model: str = "qwen-vl-max-latest",
                          is_local: bool = False) -> str:
        """图像理解功能"""
        try:
            if is_local:
                # 本地图像需要Base64编码
                base64_image = self.encode_image(image_url)
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            else:
                # 网络图像直接使用URL
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"图像理解出错: {str(e)}"
    
    def multi_image_analysis(self, 
                           image_urls: List[str], 
                           text_prompt: str,
                           model: str = "qwen-vl-max-latest") -> str:
        """多图像分析"""
        try:
            content = []
            
            # 添加所有图像
            for url in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            
            # 添加文本提示
            content.append({"type": "text", "text": text_prompt})
            
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"多图像分析出错: {str(e)}"
    
    def video_understanding(self, 
                          video_frames: List[str], 
                          text_prompt: str,
                          model: str = "qwen-vl-max-latest",
                          fps: float = 2.0) -> str:
        """视频理解（通过图像列表）"""
        try:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_frames
                        },
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"视频理解出错: {str(e)}"
    
    def video_file_understanding(self, 
                               video_url: str, 
                               text_prompt: str,
                               model: str = "qwen-vl-max-latest") -> str:
        """视频文件理解"""
        try:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url}
                        },
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"视频文件理解出错: {str(e)}"
    
    def stream_chat(self, 
                   message: str,
                   model: str = "qwen-vl-max-latest",
                   system_prompt: str = "You are a helpful assistant."):
        """流式对话"""
        try:
            stream = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                stream=True
            )
            
            print("流式输出：")
            full_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_content += content
            print("\n")
            return full_content
        except Exception as e:
            return f"流式对话出错: {str(e)}"
    
    def multi_turn_conversation(self, 
                              conversations: List[Dict[str, Any]],
                              model: str = "qwen-vl-max-latest") -> str:
        """多轮对话"""
        try:
            messages = []
            for conv in conversations:
                role = conv.get("role", "user")
                content = conv.get("content")
                
                if isinstance(content, str):
                    # 纯文本消息
                    messages.append({"role": role, "content": content})
                else:
                    # 多模态消息
                    messages.append({"role": role, "content": content})
            
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"多轮对话出错: {str(e)}"


def main():
    """主函数：展示各种API调用示例"""
    client = Qwen25VLClient()
    
    print("=" * 60)
    print("Qwen2.5-VL API调用示例")
    print("=" * 60)
    
    # 1. 纯文本对话
    print("\n1. 纯文本对话示例：")
    text_response = client.text_chat("你好，请介绍一下Qwen2.5-VL模型的主要功能。")
    print(f"回答：{text_response}")
    
    # 2. 图像理解（使用网络图片）
    print("\n2. 图像理解示例：")
    image_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
    image_response = client.image_understanding(
        image_url, 
        "请详细描述这张图片的内容，包括人物、动物、场景等。"
    )
    print(f"图像分析：{image_response}")
    
    # 3. 多图像分析
    print("\n3. 多图像分析示例：")
    image_urls = [
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
        "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"
    ]
    multi_image_response = client.multi_image_analysis(
        image_urls,
        "比较这两张图片的内容，说明它们的共同点和不同点。"
    )
    print(f"多图像分析：{multi_image_response}")
    
    # 4. 视频理解（图像序列）
    print("\n4. 视频理解示例（图像序列）：")
    video_frames = [
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/xzsgiz/football1.jpg",
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/tdescd/football2.jpg",
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/zefdja/football3.jpg",
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/aedbqh/football4.jpg"
    ]
    video_response = client.video_understanding(
        video_frames,
        "描述这个视频序列中发生的动作和过程。"
    )
    print(f"视频分析：{video_response}")
    
    # 5. 流式输出示例
    print("\n5. 流式输出示例：")
    stream_response = client.stream_chat(
        "请写一首关于人工智能的七言律诗。"
    )
    
    # 6. 多轮对话示例
    print("\n6. 多轮对话示例：")
    conversations = [
        {
            "role": "system",
            "content": "你是一个专业的图像分析师。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": "这张图片中有什么动物？"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "这张图片中有一只狗，看起来像是金毛猎犬或拉布拉多犬。"
        },
        {
            "role": "user",
            "content": "这只狗在做什么？"
        }
    ]
    
    multi_turn_response = client.multi_turn_conversation(conversations)
    print(f"多轮对话回答：{multi_turn_response}")
    
    print("\n=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)


def demo_with_local_image():
    """本地图像处理示例"""
    client = Qwen25VLClient()
    
    # 检查是否有本地图像文件
    local_image_path = "test_image.jpg"  # 替换为您的本地图像路径
    
    if os.path.exists(local_image_path):
        print(f"\n本地图像分析示例（{local_image_path}）：")
        response = client.image_understanding(
            local_image_path,
            "请分析这张图片，描述其中的内容。",
            is_local=True
        )
        print(f"本地图像分析：{response}")
    else:
        print(f"本地图像文件 {local_image_path} 不存在，跳过本地图像分析示例。")


if __name__ == "__main__":
    # 运行主要示例
    main()
    
    # 运行本地图像示例（如果有本地图像文件）
    demo_with_local_image() 