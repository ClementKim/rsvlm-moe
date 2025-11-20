import os
import json
from google import genai
from google.genai import types
from PIL import Image

# 1. 환경 설정 (API Key)
# client = genai.Client() # 환경 변수 설정 시

# 2. 모델 출력 형식을 강제하기 위한 DOTA OBB JSON 스키마
DOTA_OBB_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "detections": types.Schema(
            type=types.Type.ARRAY,
            description="A list of all detected objects with Oriented Bounding Boxes.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "box_8_coords": types.Schema(
                        type=types.Type.ARRAY,
                        description="8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4) defining the oriented bounding box in normalized (0-1000) integer values, following the DOTA convention.",
                        items=types.Schema(type=types.Type.INTEGER)
                    ),
                    "label": types.Schema(
                        type=types.Type.STRING,
                        description="The name of the detected object."
                    )
                },
                required=["box_8_coords", "label"]
            )
        )
    },
    required=["detections"]
)

# 3. 객체 감지 요청 함수 (DOTA OBB 형식)
def detect_objects_with_dota_obb(image_path, client, schema):
    """
    Gemini 모델을 사용하여 이미지에서 객체를 감지하고 DOTA OBB와 유사한 JSON 형식으로 경계 상자를 반환합니다.
    """
    try:
        if not os.path.exists(image_path):
             print(f"오류: 이미지 파일 '{image_path}'을(를) 찾을 수 없습니다.")
             return

        img = Image.open(image_path)
        
        # 모델에게 DOTA 스타일의 OBB 형식을 따르도록 명시적으로 지시하는 프롬프트
        detection_prompt = (
            "Identify all distinct objects in the image. For each object, return an oriented bounding box (OBB) defined by 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4) "
            "representing the four corner vertices in a clockwise or counter-clockwise order, using normalized integer values from 0 to 1000. "
            "The output must strictly follow the provided JSON schema."
        )
        
        response = client.generate_content(
            model='gemini-2.5-flash',
            contents=[img, detection_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        
        print("--- Gemini API 응답 (DOTA OBB JSON 형식) ---")
        # 응답이 유효한 JSON 문자열인지 확인 후 파싱
        try:
            json_output = json.loads(response.text)
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("응답 파싱 오류. 원본 텍스트를 출력합니다:")
            print(response.text)

    except Exception as e:
        print(f"오류 발생: {e}")

# 4. 실행 (실제 사용 시 API Key와 이미지 경로를 설정해야 합니다)
# IMAGE_PATH = "path/to/your/aerial_image.jpg" 
# detect_objects_with_dota_obb(IMAGE_PATH, client, DOTA_OBB_SCHEMA)