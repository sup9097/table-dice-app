from flask import Flask, render_template_string, request, session, redirect, url_for
from dice import TableDicePredictor

app = Flask(__name__)
app.secret_key = 'securekey123'
predictor = TableDicePredictor()
predictor.last_prediction = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>다이사이 예측기</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Nanum Gothic', sans-serif; background: #f9f7f3; }
        .container { max-width: 480px; margin-top: 30px; padding: 15px; background: #fff8f4; border-radius: 12px; box-shadow: 0 0 12px rgba(0,0,0,0.1); }
        .result-box { background: #fff; padding: 15px; border-radius: 10px; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); margin-top: 15px; }
        .prob-line { font-size: 0.9em; color: #5a5a5a; }
        h2 { color: #b27300; text-align: center; margin-bottom: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h2>🎲 다이사이 예측기 (머신러닝 기반)</h2>
    <form method="post" class="mb-3">
        {% if not session.get('authenticated') %}
        <div class="mb-3">
            <label class="form-label">비밀번호</label>
            <input type="password" class="form-control" name="password" required>
        </div>
        {% endif %}
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
            <input type="text" class="form-control" name="input">
        </div>
        <button type="submit" class="btn btn-warning w-100">입력 + 다음 예측</button>
    </form>
    <div class="d-flex justify-content-between mb-3">
        <form method="post" action="/reset" class="w-50 me-1">
            <button type="submit" class="btn btn-danger w-100">테이블 초기화</button>
        </form>
        <form method="post" action="/simulate" class="w-50 ms-1">
            <button type="submit" class="btn btn-success w-100">패턴 시뮬레이션</button>
        </form>
    </div>
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
    if request.method == 'POST' and 'password' in request.form:
        if not session.get('authenticated'):
            password = request.form.get('password')
            if password != "4265":
                result = "❌ 잘못된 비밀번호입니다."
                return render_template_string(HTML_TEMPLATE, tables=get_tables(), current=predictor.current_table, result=result)
            else:
                session['authenticated'] = True

    if request.method == 'POST' and 'input' in request.form:
        table = request.form['table']
        user_input = request.form['input']
        predictor.set_table(table)

        actual = tuple(sorted(map(int, list(user_input)[-3:])))
        actual_sum = sum(actual)

        if predictor.last_prediction:
            predicted_sum = sum(predictor.last_prediction)
            diff = abs(predicted_sum - actual_sum)
            reason = "✅ 정확한 예측입니다." if predicted_sum == actual_sum else f"❌ 오차 발생 (실제 합: {actual_sum})"
            if diff >= 3:
                reason += " — 차이가 큰 이유: 최근 패턴 변화 또는 데이터 부족"
            if not hasattr(predictor, 'accuracy_log'):
                predictor.accuracy_log = []
            predictor.accuracy_log.append(predicted_sum == actual_sum)
        else:
            reason = "ℹ️ 첫 입력입니다. 비교할 예측값이 없습니다."
            predicted_sum = None

        predictor.add_input(user_input)
        predictor.train_model()
        last = [predictor.histories[predictor.current_table][-1]]
        model = predictor.models[predictor.current_table]

        ml_pred = model.predict(last)[0]
        ml_pred = tuple(int(x) for x in ml_pred)
        predictor.last_prediction = ml_pred
        pred_sum = sum(ml_pred)

        probas = model.predict_proba(last)
        proba_text = ""
        for i, p in enumerate(probas):
            dist = " ".join([f"{j+1}({p[0][j]*100:.1f}%)" for j in range(len(p[0]))])
            proba_text += f"<li class='list-group-item prob-line'>🎲 타이자 {i+1}: {dist}</li>"

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

@app.route('/reset', methods=['POST'])
def reset_table():
    predictor.histories[predictor.current_table] = []
    predictor.models[predictor.current_table] = None
    predictor.last_prediction = None
    predictor.accuracy_log = []
    return redirect(url_for('home'))

@app.route('/simulate', methods=['POST'])
def simulate_pattern():
    predictor.set_table('sim2')
    for _ in range(10):
        predictor.add_input('123456')
    predictor.train_model()
    return redirect(url_for('home'))

def get_tables():
    return ["2f", "1-1", "1-2", "1-3", "1-4", "to", "sim2", "sim1-1"]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)