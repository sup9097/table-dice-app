from flask import Flask, render_template_string, request
from dice import TableDicePredictor

app = Flask(__name__)
predictor = TableDicePredictor()
predictor.last_prediction = None  # 마지막 예측값 저장용

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>다이사이 예측기</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Nanum Gothic', sans-serif; background: #f8f9fa; }
        .container { max-width: 650px; margin-top: 40px; }
        .result-box { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .prob-line { font-size: 0.9em; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="mb-4 text-primary">🎲 다이사이 예측기 (머신러닝 기반)</h2>
        <form method="post" class="mb-3">
            <div class="mb-3">
                <label class="form-label">비밀번호</label>
                <input type="password" class="form-control" name="password" required>
            </div>
            <div class="mb-3">
                <label class="form-label">테이블 선택</label>
                <select class="form-select" name="table">
                    {% for t in tables %}
                        <option value="{{ t }}" {% if t == current %}selected{% endif %}>{{ t }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">주사위 입력 (예: 123456)</label>
                <input type="text" class="form-control" name="input" required>
            </div>
            <button type="submit" class="btn btn-primary">입력 + 다음 예측</button>
        </form>

        {% if result %}
            <div class="result-box">
                {{ result|safe }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        password = request.form.get('password')
        if password != "4265":
            result = "❌ 잘못된 비밀번호입니다."
            return render_template_string(HTML_TEMPLATE, tables=get_tables(), current=predictor.current_table, result=result)

        table = request.form['table']
        user_input = request.form['input']
        predictor.set_table(table)

        # 사용자가 입력한 최신 실제값
        actual = tuple(sorted(map(int, list(user_input)[-3:])))
        actual_sum = sum(actual)

        # 이전 예측값과 비교
        if predictor.last_prediction:
            predicted_sum = sum(predictor.last_prediction)
            diff = abs(predicted_sum - actual_sum)
            reason = "✅ 정확한 예측입니다." if predicted_sum == actual_sum else f"❌ 오차 발생 (실제 합: {actual_sum})"
            if diff >= 3:
                reason += f" — 차이가 큰 이유: 최근 패턴 급변 가능성, 데이터 부족"
            if not hasattr(predictor, 'accuracy_log'):
                predictor.accuracy_log = []
            predictor.accuracy_log.append(predicted_sum == actual_sum)
        else:
            reason = "ℹ️ 첫 입력입니다. 비교할 예측값이 없습니다."
            predicted_sum = None

        # 입력 반영 및 모델 재학습
        predictor.add_input(user_input)
        predictor.train_model()
        last = [predictor.histories[predictor.current_table][-1]]
        model = predictor.models[predictor.current_table]

        # 새로운 예측값 저장
        ml_pred = model.predict(last)[0]
        ml_pred = tuple(int(x) for x in ml_pred)
        predictor.last_prediction = ml_pred
        pred_sum = sum(ml_pred)

        # 신뢰도 계산
        probas = model.predict_proba(last)
        proba_text = ""
        for i, p in enumerate(probas):
            dist = " ".join([f"{j+1}({p[0][j]*100:.1f}%)" for j in range(6)])
            proba_text += f"<li class='list-group-item prob-line'>🎲 <strong>{i+1}번 주사위:</strong> {dist}</li>"

        # 누적 정확도
        log = getattr(predictor, 'accuracy_log', [])
        total = len(log)
        correct = sum(log)
        acc_percent = (correct / total) * 100 if total else 0

        result = f"""
        <ul class="list-group mb-3">
            <li class="list-group-item">🎯 <strong>방금 입력:</strong> {actual} <span class="text-muted">(합: {actual_sum})</span></li>
            <li class="list-group-item">🤖 <strong>이전 예측:</strong> {predictor.last_prediction if predicted_sum else 'N/A'} <span class="text-muted">(합: {predicted_sum if predicted_sum else '-'})</span></li>
            <li class="list-group-item">🔍 <strong>비교 결과:</strong> {reason}</li>
            <li class="list-group-item">🔮 <strong>다음 예측값:</strong> {ml_pred} <span class="text-muted">(합: {pred_sum})</span></li>
            {proba_text}
        </ul>
        <p class="text-end text-secondary">
            📊 <strong>누적 정확도:</strong> {correct} / {total}회 ({acc_percent:.2f}%)
        </p>
        """

    return render_template_string(HTML_TEMPLATE, tables=get_tables(), current=predictor.current_table, result=result)

def get_tables():
    return ["2f", "1-1", "1-2", "1-3", "1-4", "to"]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
