"""Metadata and Session Storage for FairLens."""
import sqlite3
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = Path(os.getenv("FAIRLENS_DB_PATH", str(PROJECT_ROOT / "fairlens.db")))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            model_type TEXT NOT NULL,
            classes TEXT,
            expected_features TEXT,
            has_proba INTEGER NOT NULL,
            pipeline_steps TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            total_columns INTEGER NOT NULL,
            columns_meta TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audits (
            audit_id TEXT PRIMARY KEY,
            model_id TEXT,
            dataset_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_model_meta(model_id: str, filename: str, file_path: str, model_type: str, 
                    classes: List[Any], expected_features: Optional[List[str]], 
                    has_proba: bool, pipeline_steps: Optional[List[str]]):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO models 
        (model_id, filename, file_path, model_type, classes, expected_features, has_proba, pipeline_steps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_id, filename, file_path, model_type,
        json.dumps(classes), json.dumps(expected_features) if expected_features else None,
        1 if has_proba else 0, json.dumps(pipeline_steps) if pipeline_steps else None
    ))
    conn.commit()
    conn.close()

def get_model_meta(model_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    data["classes"] = json.loads(data["classes"]) if data["classes"] else []
    data["expected_features"] = json.loads(data["expected_features"]) if data["expected_features"] else None
    data["has_proba"] = bool(data["has_proba"])
    data["pipeline_steps"] = json.loads(data["pipeline_steps"]) if data["pipeline_steps"] else None
    return data

def save_dataset_meta(dataset_id: str, filename: str, file_path: str, total_rows: int, total_columns: int, columns_meta: List[Dict[str, Any]]):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO datasets 
        (dataset_id, filename, file_path, total_rows, total_columns, columns_meta)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        dataset_id, filename, file_path, total_rows, total_columns, json.dumps(columns_meta)
    ))
    conn.commit()
    conn.close()

def get_dataset_meta(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    data["columns_meta"] = json.loads(data["columns_meta"])
    return data

def save_audit(audit_id: str, model_id: Optional[str], dataset_id: str, config: Dict[str, Any], summary: Dict[str, Any]):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO audits (audit_id, model_id, dataset_id, config_json, summary_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        audit_id, model_id, dataset_id, json.dumps(config), json.dumps(summary)
    ))
    conn.commit()
    conn.close()

def get_audit(audit_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audits WHERE audit_id = ?", (audit_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    data["config"] = json.loads(data["config_json"])
    data["summary"] = json.loads(data["summary_json"])
    return data
