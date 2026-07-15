from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any, Dict, List

class CSVLogger:
    def __init__(self,path: str)->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fieldnames=None
    def log(self,record: Dict[str,Any])->None:
        if self.fieldnames is None: self.fieldnames=list(record.keys())
        header=not self.path.exists()
        with self.path.open("a",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=self.fieldnames)
            if header: writer.writeheader()
            writer.writerow(record)

class JSONLogger:
    def __init__(self,path: str)->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.records: List[Dict[str,Any]]=[]
        if self.path.exists():
            self.records=json.loads(self.path.read_text(encoding="utf-8"))
    def log(self,record: Dict[str,Any])->None:
        self.records.append(record)
        self.path.write_text(json.dumps(self.records,indent=2),encoding="utf-8")
