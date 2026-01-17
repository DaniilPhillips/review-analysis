"""
Аналитический модуль системы

Функции для анализа влияния отзывов на репутацию субъектов:
1. Агрегация репутации по субъектам
2. Анализ трендов
3. Сравнительный анализ
4. Выявление проблемных областей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODELS_DIR, LOGS_DIR, REPUTATION_CLASSES


class ReputationAnalytics:
    """Аналитика репутации на основе отзывов"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def load_model(self):
        """Загрузка обученной модели"""
        from model import ReputationModel
        with open(MODELS_DIR / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        self.model = ReputationModel(self.preprocessor.get_vocab_size(), 
                                      self.preprocessor.get_num_categories())
        self.model.load()
        
    def aggregate_by_subject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация метрик по субъектам репутации"""
        agg = df.groupby(["subject_id", "subject_name"]).agg({
            "rating": ["mean", "std", "count"],
            "sentiment_class": lambda x: (x == 2).mean(),  # доля позитивных
            "reputation_index": ["mean", "std"]
        }).round(3)
        agg.columns = ["avg_rating", "rating_std", "review_count", 
                       "positive_ratio", "avg_reputation", "reputation_std"]
        return agg.reset_index().sort_values("avg_reputation", ascending=False)
    
    def aggregate_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация по категориям"""
        agg = df.groupby("category").agg({
            "rating": "mean",
            "sentiment_class": lambda x: (x == 2).mean(),
            "reputation_index": "mean",
            "review_text": "count"
        }).round(3)
        agg.columns = ["avg_rating", "positive_ratio", "avg_reputation", "review_count"]
        return agg.reset_index().sort_values("avg_reputation", ascending=False)
    
    def find_problematic_areas(self, df: pd.DataFrame, threshold: float = -0.3) -> pd.DataFrame:
        """Выявление проблемных субъектов с низкой репутацией"""
        subject_stats = self.aggregate_by_subject(df)
        problematic = subject_stats[subject_stats["avg_reputation"] < threshold]
        return problematic
    
    def predict_reputation(self, texts: List[str], ratings: List[int], 
                           categories: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказание репутации для новых отзывов"""
        if self.model is None:
            self.load_model()
        
        df = pd.DataFrame({
            "review_text": texts,
            "rating": ratings,
            "category": categories,
            "subject_id": ["NEW"] * len(texts),
            "sentiment_class": [0] * len(texts),
            "reputation_index": [0] * len(texts)
        })
        X_text, X_rating, X_category, _, _ = self.preprocessor.transform(df)
        class_probs, rep_indices = self.model.predict(X_text, X_rating, X_category)
        return class_probs, rep_indices.flatten()
    
    def plot_reputation_distribution(self, df: pd.DataFrame):
        """Визуализация распределения репутации"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Распределение индекса репутации
        axes[0].hist(df["reputation_index"], bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Индекс репутации")
        axes[0].set_ylabel("Частота")
        axes[0].set_title("Распределение индекса репутации")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        
        # По классам тональности
        colors = ["#e74c3c", "#95a5a6", "#27ae60"]
        labels = ["Негативная", "Нейтральная", "Позитивная"]
        counts = df["sentiment_class"].value_counts().sort_index()
        axes[1].bar(labels, counts.values, color=colors, edgecolor="black")
        axes[1].set_ylabel("Количество отзывов")
        axes[1].set_title("Распределение по тональности")
        
        # Оценки vs репутация
        axes[2].scatter(df["rating"], df["reputation_index"], alpha=0.3)
        axes[2].set_xlabel("Оценка (1-5)")
        axes[2].set_ylabel("Индекс репутации")
        axes[2].set_title("Корреляция оценки и репутации")
        
        plt.tight_layout()
        plt.savefig(LOGS_DIR / "reputation_analysis.png", dpi=150)
        plt.close()
        print(f"Сохранено: {LOGS_DIR / 'reputation_analysis.png'}")
    
    def plot_subject_comparison(self, df: pd.DataFrame, top_n: int = 10):
        """Сравнение топ-N субъектов"""
        subject_stats = self.aggregate_by_subject(df)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Лучшие
        top = subject_stats.head(top_n)
        axes[0].barh(top["subject_name"], top["avg_reputation"], color="#27ae60")
        axes[0].set_xlabel("Средний индекс репутации")
        axes[0].set_title(f"Топ-{top_n} лучших субъектов")
        axes[0].invert_yaxis()
        
        # Худшие
        bottom = subject_stats.tail(top_n)
        axes[1].barh(bottom["subject_name"], bottom["avg_reputation"], color="#e74c3c")
        axes[1].set_xlabel("Средний индекс репутации")
        axes[1].set_title(f"Топ-{top_n} худших субъектов")
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(LOGS_DIR / "subject_comparison.png", dpi=150)
        plt.close()
        print(f"Сохранено: {LOGS_DIR / 'subject_comparison.png'}")
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Генерация текстового отчёта"""
        subject_stats = self.aggregate_by_subject(df)
        category_stats = self.aggregate_by_category(df)
        
        report = []
        report.append("=" * 60)
        report.append("АНАЛИТИЧЕСКИЙ ОТЧЁТ: ВЛИЯНИЕ ОТЗЫВОВ НА РЕПУТАЦИЮ")
        report.append("=" * 60)
        report.append(f"\nОбщая статистика:")
        report.append(f"   Всего отзывов: {len(df)}")
        report.append(f"   Субъектов: {df['subject_id'].nunique()}")
        report.append(f"   Категорий: {df['category'].nunique()}")
        
        report.append(f"\nРаспределение тональности:")
        for cls, name in REPUTATION_CLASSES.items():
            count = (df["sentiment_class"] == cls).sum()
            pct = count / len(df) * 100
            report.append(f"   {name}: {count} ({pct:.1f}%)")
        
        report.append(f"\nТоп-5 субъектов по репутации:")
        for _, row in subject_stats.head(5).iterrows():
            report.append(f"   {row['subject_name']}: {row['avg_reputation']:.3f}")
        
        report.append(f"\n⚠Проблемные субъекты (репутация < -0.3):")
        problematic = self.find_problematic_areas(df)
        if len(problematic) == 0:
            report.append("   Нет проблемных субъектов")
        else:
            for _, row in problematic.head(5).iterrows():
                report.append(f"   {row['subject_name']}: {row['avg_reputation']:.3f}")
        
        report.append(f"\nРейтинг категорий:")
        for _, row in category_stats.iterrows():
            report.append(f"   {row['category']}: {row['avg_reputation']:.3f}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """Демонстрация аналитики"""
    analytics = ReputationAnalytics()
    
    # Загрузка данных
    df = pd.read_csv(DATA_DIR / "synthetic_dataset.csv")
    
    # Генерация отчёта
    report = analytics.generate_report(df)
    print(report)
    
    # Визуализации
    analytics.plot_reputation_distribution(df)
    analytics.plot_subject_comparison(df)
    
    # Сохранение отчёта
    with open(LOGS_DIR / "analytics_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nОтчёт сохранён: {LOGS_DIR / 'analytics_report.txt'}")


if __name__ == "__main__":
    main()
