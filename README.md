# 성향 프로필 기반 관계 상담 챗봇

설문으로 만든 성향 프로필과 문서 검색 결과를 대화에 반영하는 Streamlit 프로토타입입니다. 설문·점수 시각화·검색·응답·대화 요약을 하나의 화면 흐름으로 연결합니다.

## 구현한 흐름

1. 1~7점 응답을 수집하고 역채점·축별 점수를 계산합니다.
2. 자기/타인 모형, 감정 표현, 자기효능감 조합으로 유형 설명을 표시합니다.
3. `persona_rules.json`에서 프로필에 맞는 응답 규칙을 선택합니다.
4. Chroma의 프로필·상담 가이드·위험 대응 컬렉션을 검색합니다.
5. 사용자 발화와 요약을 바탕으로 답변하고 대화 요약을 갱신합니다.

유형 분류와 위험 키워드는 이 앱의 규칙이며 임상 진단이나 검증된 위험 탐지 모델을 뜻하지 않습니다. 치료 효과·탐지 정확도·상담 효용을 측정한 결과는 없습니다.

## 기술과 자료

Python, Streamlit, Matplotlib, LangChain, Chroma, OpenAI API를 사용합니다. 현재 코드의 모델 설정은 `text-embedding-3-large`와 `gpt-5-mini`입니다. 실제 호출 호환성·응답 품질·비용은 별도 검증 대상입니다.

```text
app.py                 설문·프로필·검색·대화 화면
data/                  유형·페르소나·상담·위험 대응 JSON
assets/                표시용 폰트
requirements.txt       실행 의존성
```

## 실행과 현재 제약

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

상담 실행에는 `OPENAI_API_KEY`와 다음 Chroma 인덱스가 필요합니다.

```text
chroma_store/user_profile
chroma_store/counsel_db
chroma_store/risk_protocol
```

**현재 저장소에는 인덱스 생성용 ingest 스크립트와 저장된 벡터DB가 없습니다.** 따라서 위 명령만으로 모든 상담 기능의 재현을 보장하지 않습니다. 기존 README에서 언급했던 적재 절차는 실제 스크립트가 보완돼야 합니다. 이번 검토는 코드·구조 확인과 문서 정리이며 API 생성이나 배포 상태를 새로 검증하지 않았습니다.

`.streamlit/secrets.toml`의 `APP_PASSWORD`가 설정되면 접근 비밀번호를 사용합니다. API 키·비밀번호·개인 설문·대화 기록은 공개 저장소에 넣지 않습니다.

## 관련 프로젝트와 다음 단계

[설문·프로필 프로토타입](https://github.com/jinukjjang1129-ship-it/survey)은 설문과 시각화 부분을 별도로 살펴볼 수 있습니다. 두 저장소의 분류 기준은 같다고 가정하지 않습니다.

후속 과제는 적재 코드 복원, 자료 출처·이용 조건 정리, 모델 파라미터 호환성, 근거 기반 답변과 실패 사례 평가입니다. 개인/팀 기여의 파일별 범위는 별도 증빙 없이 확대해 기재하지 않습니다.

[기존 배포 주소](https://attachment-profile-chatbot-qqd6sxueqytzalxsrwghba.streamlit.app/) — 현재 가동 여부와 접근 권한은 별도 확인이 필요합니다.
