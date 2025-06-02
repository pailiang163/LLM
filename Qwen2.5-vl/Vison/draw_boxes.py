import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_image(image_path, json_data):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 将OpenCV图片转换为PIL格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # 设置字体参数
    try:
        # 尝试使用系统中文字体
        font_large = ImageFont.truetype("simhei.ttf", 40)  # 状态字体
        font_normal = ImageFont.truetype("simhei.ttf", 24)  # 标签字体
    except:
        try:
            # Windows系统字体
            font_large = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 40)
            font_normal = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 24)
        except:
            try:
                # 微软雅黑字体
                font_large = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 40)
                font_normal = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
            except:
                # 如果都找不到，使用默认字体
                font_large = ImageFont.load_default()
                font_normal = ImageFont.load_default()
    
    text_color = (255, 0, 0)  # 红色 (RGB格式)
    required_box_color = (0, 0, 255)   # 蓝色 - 用于required_signs
    contaminant_box_color = (255, 165, 0)   # 橙色 - 用于contaminants
    
    # 在左上角绘制状态
    status = json_data.get('status', '')
    draw.text((50, 50), status, font=font_large, fill=text_color)
    
    # 处理required_signs - 使用蓝色框
    required_signs = json_data.get('required_signs', {})
    print(f"处理required_signs: {len(required_signs)}个项目")
    for sign_name, sign_info in required_signs.items():
        print(f"  - {sign_name}: 状态={sign_info['状态']}")
        if sign_info['状态'] == '存在':
            coords = sign_info['坐标']
            x1, y1, x2, y2 = coords
            # 绘制蓝色框
            draw.rectangle([x1, y1, x2, y2], outline=required_box_color, width=3)
            # 绘制标签
            draw.text((x1, y1-30), f"[必需]{sign_name}", font=font_normal, fill=text_color)
            print(f"    绘制框: ({x1}, {y1}) -> ({x2}, {y2})")
    
    # 处理contaminants - 使用橙色框
    contaminants = json_data.get('contaminants', [])
    print(f"处理contaminants: {len(contaminants)}个项目")
    for i, contaminant in enumerate(contaminants):
        original_type = contaminant.get('类型', '未知类型')
        contaminant_status = contaminant.get('状态', '未知状态')
        print(f"  - 污染物{i+1}: {original_type}, 状态={contaminant_status}")
        
        if contaminant_status == '存在':
            coords = contaminant['坐标']
            x1, y1, x2, y2 = coords
            # 绘制橙色框
            draw.rectangle([x1, y1, x2, y2], outline=contaminant_box_color, width=3)
            # 在图片上统一显示为"其他粘贴物"，不显示描述
            display_label = "[其他粘贴物]"
            draw.text((x1, y1-30), display_label, font=font_normal, fill=text_color)
            print(f"    绘制框: ({x1}, {y1}) -> ({x2}, {y2}), 原类型: {original_type}, 显示: 其他粘贴物")
        else:
            print(f"    跳过绘制: 状态为{contaminant_status}")
    
    # 将PIL图片转换回OpenCV格式
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 保存结果图片
    output_path = image_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(output_path, result_image)
    return output_path

if __name__ == "__main__":
    # 示例使用
    image_path = r"F:\XINZAILING\Qwen_reg\094ce015646d45ba952ef3d6a0d35a7.jpg"
    json_data = {
  "status": "不干净",
  "required_signs": {
    "特种设备使用标志": {
      "坐标": [
        320,
        140,
        540,
        360
      ],
      "状态": "存在"
    },
    "乘用电梯安全注意事项": {
      "坐标": [
        320,
        360,
        540,
        480
      ],
      "状态": "缺失"
    }
  },
  "contaminants": [
    {
      "类型": "广告/违规安全注意事项/其他标识牌",
      "坐标": [
        320,
        140,
        540,
        360
      ],
      "状态": "存在",
      "描述": "电梯维保公示牌，包含维保时间、维保人员姓名、维保单位名称等信息"
    },
    {
      "类型": "广告/违规安全注意事项/其他标识牌",
      "坐标": [
        57,
        290,
        210,
        480
      ],
      "状态": "存在",
      "描述": "电子显示屏，显示外部建筑画面和部分文字信息"
    }
  ]
}


    
    try:
        output_path = draw_boxes_on_image(image_path, json_data)
        print(f"处理完成，结果已保存至: {output_path}")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}") 