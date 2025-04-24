from flask import Flask, render_template_string, request, redirect, session
from dice import TableDicePredictor
import random

app = Flask(__name__)
app.secret_key = "your-secret-key"
predictor = TableDicePredictor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>다이사이 예측기</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Nanum Gothic', sans-serif; background: #f8f9fa; }
        .container { max-width: 700px; margin-top: 40px; }
        .result-box { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="mb-4 text-primary">🎲 다이사이 예측기 (Flask)</h2>

        {% if not session.get('authenticated') %}
        <form method="post">
            <input type="hidden" name="form_type" value="login">
            <div class="mb-3">
                <label class="form-label">비밀번호</label>
                <input type="password" class="form-control" name="password" required>
            </div>
            <button type="submit" class="btn btn-primary">로그인</button>
        </form>
        {% else %}

        <form method="post" class="mb-3">
            <input type="hidden" name="form_type" value="predict">
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
            <button type="submit" class="btn btn-success">🎯 추가 + 예측</button>
        </form>

        <form method="post" class="mb-3">
            <input type="hidden" name="form_type" value="simulate">
            <div class="mb-3">
                <label class="form-label">테이블 선택 (시뮬레이션용)</label>
                <select class="form-select" name="table">
                    {% for t in tables %}
                        <option value="{{ t }}" {% if t == current %}selected{% endif %}>{{ t }}</option>
                    {% endfor %}
                </select>
                <label class="form-label mt-2">횟수</label>
                <input type="number" class="form-control" name="count" value="100" min="1" max="1000">
            </div>
            <button type="submit" class="btn btn-warning">🧪 시뮬레이션 실행</button>
        </form>

        <form method="post" class="mb-3">
            <input type="hidden" name="form_type" value="reset">
            <input type="hidden" name="table" value="{{ current }}">
            <button type="submit" class="btn btn-danger">🔄 {{ current }} 테이블 데이터 초기화</button>
        </form>

        {% if result %}
            <div class="result-box mt-4">
                <h5 class="text-success">📈 결과</h5>
                <p>{{ result|safe }}</p>
            </div>
        {% endif %}

        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    tables = ["2f", "1-1", "1-2", "1-3", "1-4", "to", "sim"]
    if "authenticated" not in session:
        session["authenticated"] = False

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "login":
            if request.form.get("password") == "4265":
                session["authenticated"] = True
            else:
                result = "❌ 잘못된 비밀번호입니다."

        elif form_type == "reset":
            table = request.form.get("table")
            predictor.histories[table] = []
            predictor.models[table] = None
            if hasattr(predictor, 'accuracy_log'):
                predictor.accuracy_log = []
            result = f"🔄 '{table}' 테이블 데이터 초기화 완료!"

        elif form_type == "predict":
            table = request.form['table']
            user_input = request.form['input']
            predictor.set_table(table)
            predictor.add_input(user_input)
            predictor.train_model()
            freq_pred = predictor.predict_next()
            last = [predictor.histories[table][-1]]
            if table in predictor.models and predictor.models[table]:
                ml_pred = predictor.models[table].predict(last)[0]
                ml_pred = tuple(int(x) for x in ml_pred)
                pred_sum = sum(ml_pred)
                actual = tuple(sorted(map(int, list(user_input)[-3:])))
                actual_sum = sum(actual)
                reason = "✅ 정확한 예측입니다." if actual_sum == pred_sum else f"❌ 오차 발생 (실제 합: {actual_sum})"
                result = (
                    f"빈도 기반 예측: {freq_pred}<br>"
                    f"🤖 머신러닝 예측: {ml_pred} (합: {pred_sum})<br>"
                    f"🎯 실제 입력: {actual} (합: {actual_sum})<br>"
                    f"🔍 분석 결과: {reason}<br>"
                )
                if not hasattr(predictor, 'accuracy_log'):
                    predictor.accuracy_log = []
                predictor.accuracy_log.append(pred_sum == actual_sum)
            else:
                result = "⚠️ 데이터 부족: 머신러닝 예측 불가"

        elif form_type == "simulate":
            count = int(request.form.get("count", 100))
            table = request.form.get("table", "sim")
            predictor.set_table(table)
            correct = 0
            total_diff = 0
            for _ in range(count):
                dice = sorted([random.randint(1, 6) for _ in range(3)])
                predictor.add_input("".join(str(x) for x in dice))
                predictor.train_model()
                last = [predictor.histories[table][-1]]
                if table in predictor.models and predictor.models[table]:
                    pred = predictor.models[table].predict(last)[0]
                    pred = tuple(int(x) for x in pred)
                    actual_sum = sum(dice)
                    pred_sum = sum(pred)
                    total_diff += abs(actual_sum - pred_sum)
                    if actual_sum == pred_sum:
                        correct += 1
            avg_diff = total_diff / count
            acc = (correct / count) * 100
            result = (
                f"🧪 {count}회 시뮬레이션 완료<br>"
                f"🎯 정확히 맞춘 횟수: {correct} / {count}<br>"
                f"📊 누적 정확도: {acc:.2f}%<br>"
                f"📉 평균 오차: {avg_diff:.2f}"
            )

    current_table = predictor.current_table if hasattr(predictor, "current_table") else "2f"
    return render_template_string(
        HTML_TEMPLATE,
        tables=tables,
        current=current_table,
        result=result
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
