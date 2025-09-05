import torch  # PyTorch 텐서 연산 및 모델 학습/추론을 위한 기본 라이브러리

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments  # Hugging Face 모델/토크나이저 및 학습 유틸
from torch.utils.data import Dataset  # 사용자 정의 데이터셋 생성을 위한 베이스 클래스
from torch.optim import AdamW  # AdamW 옵티마이저 (L2 규제를 분리한 Adam)

class MenuDataset(Dataset):  # 간단한 텍스트 목록을 GPT-2 학습 입력 형식으로 변환하는 커스텀 데이터셋
    def __init__(self, tokenizer, texts, max_length = 512):  # 토크나이저, 텍스트 리스트, 최대 길이 설정
        self.tokenizer = tokenizer  # 토큰화에 사용할 토크나이저 보관
        self.inputs = []  # input_ids 텐서들을 저장할 리스트
        self.attn_masks = []  # attention_mask 텐서들을 저장할 리스트
        self.labels = []  # 언어모델 학습용 레이블(다음 토큰 예측), 여기선 입력과 동일하게 사용

        for text in texts:  # 각 텍스트에 대해
            encodings_dict = tokenizer(text, truncation = True, max_length = max_length, padding = "max_length")  # 최대 길이로 잘라내고 패딩
            self.inputs.append(torch.tensor(encodings_dict['input_ids']))  # 토큰 ID를 텐서로 저장
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))  # 패딩 구분을 위한 마스크 저장
            self.labels.append(torch.tensor(encodings_dict['input_ids']))  # LM 학습이므로 레이블을 입력과 동일하게 설정

    def __len__(self):  # 데이터셋 길이 반환
        return len(self.inputs)  # 샘플 수는 입력 리스트 길이와 동일

    def __getitem__(self, idx):  # 인덱스에 해당하는 샘플 반환
        return {
            'input_ids': self.inputs[idx],  # 모델 입력 토큰 ID
            'attention_mask': self.attn_masks[idx],  # 패딩 여부를 알리는 마스크
            'labels': self.labels[idx]  # 교사강요(다음 토큰 예측)용 레이블
        }

texts = [  # 학습에 사용할 간단한 예시 대화 데이터
    "User: Show me the menu. Waiter: Here's the menu: hamburger, cola.",  # 메뉴를 보여주는 간단한 문장
]

# 토크나이저와 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # 사전학습된 GPT-2 토크나이저 로드
# 특수 토큰 추가
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 패딩 토큰이 없는 GPT-2에 [PAD] 토큰을 추가
model = GPT2LMHeadModel.from_pretrained('gpt2')  # 텍스트 생성을 위한 LM 헤드가 달린 GPT-2 모델 로드
# 토큰 임베딩 크기 재조정
model.resize_token_embeddings(len(tokenizer))  # 토큰 수가 증가했으므로 임베딩 행렬 크기를 새 어휘 크기에 맞춤

optimizer = AdamW(model.parameters(), lr=5e-5)  # AdamW 옵티마이저로 모델 파라미터를 학습할 설정

# Ensure model is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU가 있으면 CUDA, 없으면 CPU 사용
model.to(device)  # 모델을 선택한 디바이스로 이동시켜 연산 일관성 보장

train_dataset = MenuDataset(tokenizer, texts)  # 텍스트 데이터를 토큰 ID/마스크/레이블로 구성한 데이터셋 인스턴스 생성

training_args = TrainingArguments(  # Trainer가 사용할 학습 설정 하이퍼파라미터
    output_dir='./results',  # 체크포인트 및 로그 저장 경로
    num_train_epochs=50,  # 전체 데이터셋에 대한 에폭 수
    per_device_train_batch_size=2,  # 디바이스당 배치 크기(여기선 1개 GPU)
    warmup_steps=100,  # 러닝레이트 워ーム업 스텝 수
    weight_decay=0.01,  # L2 정규화 강도
    logging_dir='./logs',  # 로그 저장 디렉토리
    logging_steps=10,  # 로그 출력 간격(스텝 단위)
)

trainer = Trainer(  # Hugging Face의 고수준 학습 루프 유틸
    model=model,  # 학습할 모델
    args=training_args,  # 학습 설정
    train_dataset=train_dataset,  # 학습 데이터셋
    optimizers=(optimizer, None)  # (optimizer, scheduler) 형태로 전달, 스케줄러는 None
)

# 학습 시작
trainer.train()  # Trainer가 내부 루프(epoch/step)로 모델 학습 수행

# 학습된 모델로 질문에 답변
prompt = "User: Show me the menu."  # 생성 시작에 사용할 프롬프트 텍스트

# 입력을 토크나이저로 처리하고 모델과 같은 디바이스로 이동
inputs = tokenizer(prompt, return_tensors='pt')  # 프롬프트를 토큰 ID와 어텐션 마스크로 변환
inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력 텐서들을 모델과 동일 디바이스(GPU/CPU)로 이동

# 답변 생성
outputs = model.generate(  # 모델의 오토레그레시브 생성 API 호출
    input_ids=inputs['input_ids'],  # 시작 시퀀스로 사용할 입력 토큰들
    attention_mask=inputs.get('attention_mask', None),  # 패딩 위치를 무시하도록 하는 마스크
    max_length=1024,  # 최대 생성 길이(주의: 기본 GPT-2의 컨텍스트 윈도는 1024로, 너무 크면 에러/성능저하 가능)
    num_return_sequences=1,  # 생성할 후보 문장 수
    no_repeat_ngram_size=2,  # 동일한 2-그램 반복 방지로 다양성 확보
    pad_token_id=tokenizer.pad_token_id,  # PAD 토큰 ID 사용(경고 방지)
    do_sample=True,  # 샘플링 기반 생성 활성화(확률적으로 다양성 부여)
    temperature=0.7,  # 분포를 평탄하게 하여 창의성 조절(낮을수록 보수적)
    top_k=50,  # 상위 K개 후보만 고려하는 샘플링
    top_p=0.95  # 누적 확률이 p를 넘는 최소 후보 집합에서 샘플링(뉴클리어스)
)

# 생성된 출력을 텍스트로 디코드
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 토큰 ID 시퀀스를 사람이 읽는 문자열로 변환
# 인코딩 문제를 해결하기 위해 UTF-8을 명시적으로 사용
print(answer.encode('utf-8').decode('utf-8'))  # 콘솔 인코딩 이슈 방지용 출력