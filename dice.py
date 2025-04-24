import random
import numpy as np
from collections import Counter, defaultdict
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

ALLOWED_TABLES = ["2f", "1-1", "1-2", "1-3", "1-4", "to", "sim-2f", "sim-1-1", "sim-1-2", "sim-1-3", "sim-1-4"]

class TableDicePredictor:
    def __init__(self):
        self.data_dir = "table_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.histories = defaultdict(list)
        self.models = {}
        self.current_table = "2f"
        self.load_all()
        self.aggregate_table("to")

    def get_table_path(self, table):
        return os.path.join(self.data_dir, f"{table}_history.json")

    def save_history(self, table):
        all_data = self.histories[table]
        existing = []
        if not all_data:
            return
        for file in os.listdir(self.data_dir):
            if file.endswith(f"_{table}_history.json"):
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    try:
                        existing.extend(json.load(f))
                    except:
                        pass
        merged = existing + all_data
        path = os.path.join(self.data_dir, f"{datetime.now().strftime('%Y-%m-%d')}_{table}_history.json")
        with open(path, 'w') as f:
            json.dump(merged, f)
        print(f"📁 {path} 저장됨 - 총 {len(merged)}개")

    def load_all(self):
        for file in os.listdir(self.data_dir):
            if file.endswith("_history.json"):
                parts = file.replace("_history.json", "").split("_", 1)
                table = parts[1] if len(parts) == 2 else parts[0]
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    self.histories[table].extend([tuple(roll) for roll in json.load(f)])

    def aggregate_table(self, target_table):
        combined = []
        for table in ["2f", "1-1", "1-2", "1-3", "1-4"]:
            combined.extend(self.histories.get(table, []))
        self.histories[target_table] = combined
        self.save_history(target_table)

    def set_table(self, name):
        if name not in ALLOWED_TABLES:
            print(f"⚠️ 사용할 수 없는 테이블 이름입니다. 허용된 테이블: {', '.join(ALLOWED_TABLES)}")
            return
        if name == "to":
            self.aggregate_table(name)
        self.current_table = name
        print(f"\n📌 현재 테이블: {name}")

    def add_input(self, input_string):
        if input_string and len(input_string) % 3 == 0:
            try:
                rolls = [tuple(sorted(map(int, input_string[i:i+3]))) for i in range(0, len(input_string), 3)]
                self.histories[self.current_table].extend(rolls)
                self.save_history(self.current_table)
            except ValueError:
                print("⚠️ 숫자 외 문자가 포함되어 있습니다.")

    def simulate_rolls(self, count=100):
        for _ in range(count):
            roll = tuple(sorted([random.randint(1, 6) for _ in range(3)]))
            self.histories[self.current_table].append(roll)
        self.save_history(self.current_table)
        print(f"🎲 {count}개 시뮬레이션 데이터 추가됨")

    def predict_next(self, top_n=2):
        history = self.histories[self.current_table]
        if not history:
            return [tuple(np.random.randint(1, 7, 3)) for _ in range(top_n)]
        counter = Counter(history)
        return [comb for comb, _ in counter.most_common(top_n)]

    def train_model(self):
        history = self.histories[self.current_table]
        if len(history) < 5:
            print("⚠️ 데이터 부족")
            return
        X = np.array(history[:-1])
        y = np.array(history[1:])
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        model.fit(X, y)
        self.models[self.current_table] = model
        print("✅ 모델 학습 완료")

    def predict_with_model(self):
        if self.current_table not in self.models:
            print("⚠️ 모델이 없습니다. 먼저 학습하세요 (m)")
            return
        last = np.array([self.histories[self.current_table][-1]])
        for i in range(2):
            noisy_last = last + np.random.randint(-1, 2, last.shape)
            prediction = self.models[self.current_table].predict(noisy_last)[0]
            number_counter = Counter(prediction)
            most_probable_number = number_counter.most_common(1)[0][0]
            prediction = tuple(int(x) for x in prediction)
            predicted_sum = sum(prediction)
            print(f"🤖 머신러닝 예측 {i+1}: {prediction} → 🔢 가장 높은 확률 숫자: {most_probable_number}")
            if predicted_sum in [16, 15, 5, 6]:
                count = 0
                for _ in range(100):
                    sample = self.models[self.current_table].predict(last)[0]
                    if sum(sample) == predicted_sum:
                        count += 1
                if count >= 50:
                    print(f"⚠️ 경고: 합이 {predicted_sum}일 확률이 {count}%로 50% 이상입니다!")

    def analyze_position_correlation(self):
        print("📊 테이블별 주사위 위치 상관관계 비교:")
        for table in [t for t in self.histories if len(self.histories[t]) >= 10]:
            history = np.array(self.histories[table])
            print(f"\n📁 테이블: {table} ({len(history)}개)")
            correlation = np.corrcoef(history.T)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        corr_score = int(round(abs(correlation[i, j]) * 10))
                        emoji = "🧊" if corr_score < 3 else ("🌡" if corr_score < 7 else "🔥")
                        print(f"  🎯 위치 {i+1} vs 위치 {j+1} 유사도 점수: {corr_score}/10 {emoji}")
        print("\n✅ 전체 테이블 분석 완료.")

    def analyze_current_table_correlation(self):
        history = np.array(self.histories[self.current_table])
        if len(history) < 10:
            print("⚠️ 데이터 부족 - 상관관계 분석 생략")
            return
        print("📊 주사위 위치별 상관관계 분석:")
        correlation = np.corrcoef(history.T)
        for i in range(3):
            for j in range(3):
                if i != j:
                    print(f"  🎯 위치 {i+1} vs 위치 {j+1} 상관계수: {correlation[i, j]:.2f}")

    def evaluate_accuracy(self):
        history = self.histories[self.current_table]
        if len(history) < 10:
            print("⚠️ 데이터 부족")
            return
        X = np.array(history[:-1])
        y = np.array(history[1:])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(3)]
        print(f"🎯 정확도: 1번={acc[0]*100:.2f}%, 2번={acc[1]*100:.2f}%, 3번={acc[2]*100:.2f}%")

    def undo_last(self):
        if self.histories[self.current_table]:
            removed = self.histories[self.current_table].pop()
            self.save_history(self.current_table)
            print(f"⏪ 마지막 입력 {removed} 삭제 완료")
        else:
            print("⚠️ 삭제할 데이터가 없습니다.")

    def reset_history(self):
        self.histories[self.current_table] = []
        self.save_history(self.current_table)
        print("🧹 테이블 기록 초기화 완료")

    def copy_current_to_sim(self):
        sim_table = f"sim-{self.current_table}" if not self.current_table.startswith("sim-") else self.current_table
        if sim_table not in ALLOWED_TABLES:
            print(f"⚠️ 시뮬레이션 테이블 '{sim_table}' 이 허용되지 않습니다.")
            return
        self.histories[sim_table].extend(self.histories[self.current_table])
        self.save_history(sim_table)
        print(f"📥 현재 테이블의 데이터를 {sim_table} 시뮬레이션 테이블로 복사 완료")

if __name__ == "__main__":
    predictor = TableDicePredictor()
    print("\n🎲 다이사이 테이블 예측기 (단축명령어 지원)")
    print(f"허용된 테이블: {', '.join(ALLOWED_TABLES)}")
    while True:
        print("""
========================================
📌 현재 선택된 테이블: {table}
========================================""".format(table=predictor.current_table))
        cmd = input("명령어 (t/i/s/p/m/a/u/r/c/h/q): ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 'h':
            print("""
🆘 명령어 도움말:
 t - 테이블 선택
 i - 주사위 숫자 입력 (예: 123456)
 s - 시뮬레이션 데이터 추가
 p - 빈도 기반 예측
 m - 모델 학습 (Random Forest)
 a - 예측 정확도 평가
 u - 마지막 입력 삭제
 r - 테이블 기록 초기화
 c - 현재 테이블 → 시뮬레이션 테이블로 복사
 q - 종료
 h - 명령어 도움말
""")
        elif cmd == 't':
            name = input("테이블 이름: ").strip()
            predictor.set_table(name)
        elif cmd == 'i':
            string = input("주사위 숫자 입력 (예: 123456): ").strip()
            predictor.add_input(string)
            print("📈 예측 결과:", predictor.predict_next())
            predictor.train_model()
            predictor.predict_with_model()
        elif cmd == 's':
            count = input("시뮬레이션 횟수: ").strip()
            predictor.simulate_rolls(int(count))
        elif cmd == 'p':
            print("📈 예측 결과:", predictor.predict_next())
        elif cmd == 'm':
            predictor.train_model()
        elif cmd == 'a':
            predictor.analyze_current_table_correlation()
            predictor.evaluate_accuracy()
            predictor.analyze_position_correlation()
        elif cmd == 'u':
            predictor.undo_last()
        elif cmd == 'r':
            predictor.reset_history()
        elif cmd == 'c':
            predictor.copy_current_to_sim()
        else:
            print("❓ 알 수 없는 명령어입니다. t/i/s/p/m/a/u/r/c/h/q 중 하나를 입력하세요")
