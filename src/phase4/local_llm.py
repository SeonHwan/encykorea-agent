"""로컬 모델(GPU)로 답변을 생성한다. ask_claude·ask_gpt와 같은 모양의 결과를 돌려준다.
프로젝트 루트에서 실행한다 (answer.py가 chunk.py를 import하기 때문)."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer import SYSTEM_PROMPT


def load_local_model(model_name, dtype=torch.bfloat16, quantize_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,  # 연산은 16비트로 하고 저장만 4비트로 한다
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda")
    model.eval()  # 학습 모드가 아니라 추론 모드 (드롭아웃 등을 끈다)
    return model, tokenizer


def ask_local(model, tokenizer, user_message, max_new_tokens=1000):
    # ask_claude·ask_gpt와 같은 시스템 프롬프트(SYSTEM_PROMPT)와 같은 사용자 메시지를 쓰고 같은 모양의 결과를 돌려준다.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    # 채팅 템플릿: 모델이 학습한 대화 형식(역할 표시 등)으로 입력을 감싼다. 모델마다 형식이 달라서 토크나이저가 만든다.
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # 이 transformers 버전은 텐서가 아니라 input_ids·attention_mask를 담은 BatchEncoding을 돌려준다 (딕셔너리처럼 꺼낸다)
    input_ids = encoded["input_ids"].to(model.device)

    started = time.time()
    with torch.no_grad():  # 기울기 계산을 끈다 (추론에는 필요 없고 메모리를 아낀다)
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 같은 입력에는 같은 답이 나오게 한다 (비교를 쉽게)
        )
    seconds = time.time() - started

    # 출력에는 입력 토큰도 앞에 붙어 있어서 입력 길이 뒤부터 잘라낸다
    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 끝 토큰(EOS) 없이 max_new_tokens에서 멈췄으면 잘린 것이다
    stop = "length" if new_tokens.shape[0] >= max_new_tokens else "stop"
    return {"text": text, "input_tokens": int(input_ids.shape[1]), "output_tokens": int(new_tokens.shape[0]),
            "stop": stop, "seconds": round(seconds, 1),
            "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
