from openai import OpenAI
import os
import json
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_elevator_image(client, image_path):
    base64_image = encode_image(image_path)
    
    system_prompt = """这是一个电梯监控画面，请你仔细分析图中内容：

观察有特种设备使用标志 和乘用电梯安全注意事项吗？

除了这两个，电梯的墙面是否还有别的广告、品牌贴纸、其他标识牌？

备注：
特种设备使用标志：
标志牌的边框可能为绿色，内容背景为白色或浅色，有盖章
它上面的内容可能包括： 标志名、内容包括设备编号、使用单位、检验机构、检验日期、二维码等信息
注意区分：电梯安全使用或管理相关的标识牌和乘用电梯安全注意事项非白色背景，也是其他标识牌

乘用电梯安全注意事项：
背景为白色，边框是红色，通常为A4纸大小或稍大,没有国徽标志，整体设计醒目，便于引起注意。
标识标题是固定的乘用电梯安全注意事项且为红色，不支持不一致的标题
文字和图标主要为黑色，部分图标使用红色以增强警示效果。
注意区分： 电梯维保公示牌（它上面的内容可能包含：标题、表格、维保时间、维保人员姓名、维保单位名称、维保日期等详细信息。）
检查背景颜色：背景不是白色，不是乘用电梯安全注意事项



输出格式：
例如：
{
  "status": "干净" | "不干净",
  "required_signs": {
    "特种设备使用标志": {"坐标": [x1, y1, x2, y2], "状态": "存在/缺失"},
    "乘用电梯安全注意事项": {"坐标": [x1, y1, x2, y2], "状态": "存在/缺失"}
  },
  "contaminants": [
    {
      "类型": "广告/违规安全注意事项/其他标识牌",
      "坐标": [x1, y1, x2, y2],"状态": "存在/缺失"，
      "描述": "具体内容（如‘蓝色背景的安全注意事项’）"
    }
  ]
}
坐标：是问题存在的位置,(x1,y1)表示左上角点坐标，(x2,y2)表示右下角点坐标,状态不是存在，不输出坐标
状态：只有特种设备使用标志和电梯安全注意事项标识牌且同时存在，干净，否则不干净。

"""
    
    response = client.chat.completions.create(
        model="qwen2.5-vl-32b-instruct",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这是一个电梯监控画面，请按照要求分析并输出JSON格式的结果"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

def process_images():
    client = OpenAI(
        api_key="sk-52918409d9904c8699a5e0765371cf22",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    os.makedirs('32bresults', exist_ok=True)
    
    with open('results/result-v3.txt', 'w', encoding='utf-8') as result_file:
        for filename in os.listdir(r'C:\Users\zhupailiang\Desktop\电梯\轿厢照片'):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(r'C:\Users\zhupailiang\Desktop\电梯\轿厢照片', filename)
                
                try:
                    print(f"Processing {filename}...")
                    response = analyze_elevator_image(client, image_path)
                    
                    try:
                        json_response = json.loads(response)
                        formatted_response = json.dumps(json_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        formatted_response = f"Invalid JSON response:\n{response}"
                    
                    result_file.write(f"图片: {filename}\n分析结果:\n{formatted_response}\n\n")
                    print(f"Successfully processed {filename}")
                    
                except Exception as e:
                    error_msg = f"处理图片 {filename} 时出错: {str(e)}"
                    result_file.write(f"{error_msg}\n\n")
                    print(error_msg)

if __name__ == '__main__':
    import time
    start = time.time()
    process_images()
    end = time.time()
    print("time: ",(end-start)*1000,'ms')
    print("所有图片处理完成，结果已保存到 results/result.txt")