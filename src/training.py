"""
Модуль обучения модели
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, mean_absolute_error
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_CONFIG, DATA_DIR, MODELS_DIR, LOGS_DIR
from preprocessing import TextPreprocessor
from model import ReputationModel


class ModelTrainer:
    """Тренер модели анализа репутации"""
    
    def __init__(self, config: Dict = None):
        self.config = config or TRAINING_CONFIG
        self.preprocessor: Optional[TextPreprocessor] = None
        self.model: Optional[ReputationModel] = None
        self.history = None
        
    def load_data(self, filepath: Path = None) -> pd.DataFrame:
        filepath = filepath or DATA_DIR / "synthetic_dataset.csv"
        df = pd.read_csv(filepath)
        print(f"✅ Загружено {len(df)} записей")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val_df, test_df = train_test_split(df, test_size=self.config["test_split"], 
                                                  random_state=42, stratify=df["sentiment_class"])
        val_ratio = self.config["validation_split"] / (1 - self.config["test_split"])
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio,
                                            random_state=42, stratify=train_val_df["sentiment_class"])
        print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def prepare_data(self, train_df, val_df, test_df) -> Dict[str, np.ndarray]:
        self.preprocessor = TextPreprocessor()
        self.preprocessor.fit(train_df["review_text"].tolist(), train_df["category"].tolist(),
                              train_df["subject_id"].tolist())
        data = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            X_text, X_rating, X_category, y_class, y_index = self.preprocessor.transform(df)
            data[f"{name}_X_text"] = X_text
            data[f"{name}_X_rating"] = X_rating
            data[f"{name}_X_category"] = X_category
            data[f"{name}_y_class"] = y_class
            data[f"{name}_y_index"] = y_index
        return data
    
    def train(self, data: Dict[str, np.ndarray]) -> 'ModelTrainer':
        self.model = ReputationModel(self.preprocessor.get_vocab_size(), self.preprocessor.get_num_categories())
        print("\n🚀 Начало обучения...")
        self.history = self.model.fit(
            X_text=data["train_X_text"], X_rating=data["train_X_rating"],
            X_category=data["train_X_category"], y_class=data["train_y_class"],
            y_index=data["train_y_index"],
            validation_data=(
                {"text_input": data["val_X_text"], "rating_input": data["val_X_rating"],
                 "category_input": data["val_X_category"]},
                {"sentiment_output": data["val_y_class"], "reputation_output": data["val_y_index"]}
            ),
            batch_size=self.config["batch_size"], epochs=self.config["epochs"], verbose=1
        )
        print("✅ Обучение завершено!")
        return self
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        print("\n📊 Оценка модели...")
        class_probs, rep_indices = self.model.predict(data["test_X_text"], data["test_X_rating"], data["test_X_category"])
        y_pred_class = np.argmax(class_probs, axis=1)
        y_true_class = data["test_y_class"]
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class, average="macro")
        mae = mean_absolute_error(data["test_y_index"], rep_indices.flatten())
        
        print(f"\n📈 Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MAE: {mae:.4f}")
        print(classification_report(y_true_class, y_pred_class, target_names=["негативная", "нейтральная", "позитивная"]))
        return {"accuracy": accuracy, "f1": f1, "mae": mae}
    
    def plot_training_history(self):
        if not self.history: return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(self.history.history["loss"], label="Train")
        axes[0,0].plot(self.history.history["val_loss"], label="Val")
        axes[0,0].set_title("Loss"); axes[0,0].legend()
        axes[0,1].plot(self.history.history["sentiment_output_accuracy"], label="Train")
        axes[0,1].plot(self.history.history["val_sentiment_output_accuracy"], label="Val")
        axes[0,1].set_title("Accuracy"); axes[0,1].legend()
        axes[1,0].plot(self.history.history["sentiment_output_loss"], label="Train")
        axes[1,0].plot(self.history.history["val_sentiment_output_loss"], label="Val")
        axes[1,0].set_title("Sentiment Loss"); axes[1,0].legend()
        axes[1,1].plot(self.history.history["reputation_output_mae"], label="Train")
        axes[1,1].plot(self.history.history["val_reputation_output_mae"], label="Val")
        axes[1,1].set_title("Reputation MAE"); axes[1,1].legend()
        plt.tight_layout()
        plt.savefig(LOGS_DIR / "training_history.png", dpi=150)
        plt.close()
    
    def plot_confusion_matrix(self, data):
        class_probs, _ = self.model.predict(data["test_X_text"], data["test_X_rating"], data["test_X_category"])
        cm = confusion_matrix(data["test_y_class"], np.argmax(class_probs, axis=1))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Негат.", "Нейтр.", "Позит."], yticklabels=["Негат.", "Нейтр.", "Позит."])
        plt.xlabel("Предсказание"); plt.ylabel("Истина"); plt.title("Confusion Matrix")
        plt.savefig(LOGS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
    
    def save_artifacts(self):
        self.model.save()
        with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)

def main():
    trainer = ModelTrainer()
    df = trainer.load_data()
    train_df, val_df, test_df = trainer.split_data(df)
    data = trainer.prepare_data(train_df, val_df, test_df)
    trainer.train(data)
    metrics = trainer.evaluate(data)
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(data)
    trainer.save_artifacts()
    return trainer, metrics

if __name__ == "__main__":
    main()
