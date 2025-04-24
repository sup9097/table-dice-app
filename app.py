from flask import Flask, render_template_string, request
from dice import TableDicePredictor

app = Flask(__name__)
predictor = TableDicePredictor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"ko\">
<head>
    <meta charset=\"UTF-8\">
    <title>다이사이 예측기</title>
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
    <style>
        body { font-family: 'Nanum Gothic', sans-serif; background: #f8f9fa; }
        .container { max-width: 600px; margin-top: 40px; }
        .result-box { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class=\"container\">
        <h2 class=\"mb-4 text-primary\">🎲 다이사이 예측기 (Flask)</h2>
        <form method=\"post\" class=\"mb-3\">
            <div class=\"mb-3\">
                <label class=\"form-label\">비밀번호</label>
                <input type=\"password\" class=\"form-control\" name=\"password\" required>
            </div>
            <div class=\"mb-3\">
                <label class=\"form-label\">테이블 선택</label>
                <select class=\"form-select\" name=\"table\">
                    {% for t in tables %}
                        <option value=\"{{ t }}\" {% if t == current %}selected{% endif %}>{{ t }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class=\"mb-3\">
                <label class=\"form-label\">주사위 입력 (예: 123456)</label>
                <input type=\"text\" class=\"form-control\" name=\"input\" required>
            </div>
            <button type=\"submit\" class=\"btn btn-primary\">추가 + 예측</button>
        </form>

        {% if result %}
            <div class=\"result-box\">
                <h5 class=\"text-success\">📈 예측 결과</h5>
                <p>{{ result|safe }}</p>
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
            return render_template_string(
                HTML_TEMPLATE,
                tables=["2f", "1-1", "1-2", "1-3", "1-4", "to"],
                current=predictor.current_table,
                result=result
            )
        table = request.form['table']
        user_input = request.form['input']
        predictor.set_table(table)
        predictor.add_input(user_input)
        predictor.train_model()
        freq_pred = predictor.predict_next()
        last = [predictor.histories[predictor.current_table][-1]]
        ml_pred = predictor.models[predictor.current_table].predict(last)[0]
        ml_pred = tuple(int(x) for x in ml_pred)
        pred_sum = sum(ml_pred)
        actual = tuple(sorted(map(int, list(user_input)[-3:])))  # 마지막 주사위 결과 추정
        actual_sum = sum(actual)

        # 이유 분석: 예측과 실제의 차이
        reason = "✅ 정확한 예측입니다." if actual_sum == pred_sum else f"❌ 오차 발생 (실제 합: {actual_sum})"
        diff = abs(pred_sum - actual_sum)
        if diff >= 3:
            reason += f" — 차이가 큰 이유: 최근 패턴 급변 가능성, 데이터 부족"

        if not hasattr(predictor, 'accuracy_log'):
            predictor.accuracy_log = []
        predictor.accuracy_log.append(abs(pred_sum - actual_sum) == 0)
    total = len(predictor.accuracy_log)
    correct = sum(predictor.accuracy_log)
    acc_percent = (correct / total) * 100 if total else 0

    result = (
        f"빈도 기반 예측: {freq_pred}<br>"
        f"🤖 머신러닝 예측: {ml_pred} (합: {pred_sum})<br>"
        f"🎯 실제 입력: {actual} (합: {actual_sum})<br>"
        f"🔍 분석 결과: {reason}<br>"
        f"📊 누적 정확도: {correct} / {total}회 ({acc_percent:.2f}%)"
    )
    return render_template_string(
        HTML_TEMPLATE,
        tables=["2f", "1-1", "1-2", "1-3", "1-4", "to"],
        current=predictor.current_table,
        result=result
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
