import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QComboBox, QLineEdit, QTextEdit, QMessageBox
)
from dice import TableDicePredictor

ALLOWED_TABLES = ["2f", "1-1", "1-2", "1-3", "1-4", "to", "sim-2f", "sim-1-1", "sim-1-2", "sim-1-3", "sim-1-4"]

class DicePredictorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = TableDicePredictor()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("다이사이 예측기 (PyQt5)")
        self.setGeometry(300, 300, 600, 400)

        self.table_combo = QComboBox()
        self.table_combo.addItems(ALLOWED_TABLES)
        self.table_combo.setCurrentText(self.predictor.current_table)
        self.table_combo.currentTextChanged.connect(self.change_table)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("예: 123456")

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        add_btn = QPushButton("추가")
        predict_btn = QPushButton("예측")
        train_btn = QPushButton("모델 학습")
        accuracy_btn = QPushButton("정확도 확인")
        simulate_btn = QPushButton("시뮬레이션")
        reset_btn = QPushButton("초기화")
        delete_btn = QPushButton("삭제")

        add_btn.clicked.connect(self.add_input)
        predict_btn.clicked.connect(self.predict)
        train_btn.clicked.connect(self.train_model)
        accuracy_btn.clicked.connect(self.evaluate)
        simulate_btn.clicked.connect(self.simulate)
        reset_btn.clicked.connect(self.reset)
        delete_btn.clicked.connect(self.undo)

        h_top = QHBoxLayout()
        h_top.addWidget(QLabel("테이블:"))
        h_top.addWidget(self.table_combo)
        h_top.addStretch()

        h_input = QHBoxLayout()
        h_input.addWidget(QLabel("입력값:"))
        h_input.addWidget(self.input_edit)
        h_input.addWidget(add_btn)

        h_buttons = QHBoxLayout()
        h_buttons.addWidget(predict_btn)
        h_buttons.addWidget(train_btn)
        h_buttons.addWidget(accuracy_btn)
        h_buttons.addWidget(simulate_btn)
        h_buttons.addWidget(reset_btn)
        h_buttons.addWidget(delete_btn)

        layout = QVBoxLayout()
        layout.addLayout(h_top)
        layout.addLayout(h_input)
        layout.addLayout(h_buttons)
        layout.addWidget(QLabel("출력 결과:"))
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def change_table(self, name):
        self.predictor.set_table(name)
        self.result_text.append(f"📌 테이블 전환: {name}")

    def add_input(self):
        val = self.input_edit.text().strip()
        self.predictor.add_input(val)
        self.result_text.append(f"➕ 입력: {val}")
        self.input_edit.clear()

    def predict(self):
        results = self.predictor.predict_next()
        self.result_text.append("📈 예측 결과: " + ", ".join(str(r) for r in results))

        # 머신러닝 예측
        if self.predictor.current_table not in self.predictor.models:
            self.predictor.train_model()

        last = [self.predictor.histories[self.predictor.current_table][-1]]
        prediction = self.predictor.models[self.predictor.current_table].predict(last)[0]
        prediction = tuple(int(x) for x in prediction)
        predicted_sum = sum(prediction)
        self.result_text.append(f"🤖 머신러닝 예측: {prediction} (합: {predicted_sum})")

    def train_model(self):
        self.predictor.train_model()
        self.result_text.append("✅ 모델 학습 완료")

    def evaluate(self):
        self.result_text.append("🎯 모델 정확도:")
        self.predictor.evaluate_accuracy()

    def simulate(self):
        self.predictor.simulate_rolls()
        self.result_text.append("🎲 시뮬레이션 완료")

    def reset(self):
        self.predictor.reset_history()

        import os
        deleted_files = []
        for file in os.listdir(self.predictor.data_dir):
            if file.endswith(f"_{self.predictor.current_table}_history.json"):
                os.remove(os.path.join(self.predictor.data_dir, file))
                deleted_files.append(file)

        self.result_text.append("🧹 초기화 완료")
        if deleted_files:
            self.result_text.append(f"🗑 삭제된 파일: {', '.join(deleted_files)}")

    def undo(self):
        self.predictor.undo_last()
        self.result_text.append("⏪ 마지막 입력 삭제")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DicePredictorGUI()
    win.show()
    sys.exit(app.exec_())
