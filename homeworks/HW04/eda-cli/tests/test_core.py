from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# Новые тесты для новых эвристик качества данных

def test_quality_flags_id_dulicates():
    """Тест для проверки дубликатов в 'id'-столбцах."""
    df = pd.DataFrame({
        'customer_id' : [1, 2, 3, 1, 2], # дубликаты
        'order_id': [100, 101, 102, 103, 104],  # нет дубликатов
        'values' : [50, 10, 20, 40, 30]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    # Проверяем, что флаг установлен правильно
    assert flags["has_suspicious_id_duplicates"] == True
    assert len(flags["id_duplicates"]) > 0
    assert "customer_id" in flags["id_duplicates"]
    assert flags["id_duplicates"]["customer_id"]["duplicates_count"] == 2
    assert flags["id_duplicates"]["customer_id"]["duplicates_share"] == 0.4
    assert flags["total_id_duplicate_share"] == 0.4
    assert "order_id" not in flags["id_duplicates"]


def test_quality_flags_constant_columns():
    """Тест для проверки константных колонок."""
    df = pd.DataFrame({
        'id' : [1, 2, 3, 4, 5, 6],
        'value' : [20, 10, 40, 50, 30, 60],
        'constant_col' : ['1', '1', '1', '1', '1', '1'] # константная
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    # Проверяем, что флаг установлен правильно
    assert flags["has_constant_columns"] == True
    assert 'constant_col' in flags['constant_columns']
    assert flags['constant_columns_count'] == 1
    assert 0.0 <= flags['quality_score'] <= 1.0