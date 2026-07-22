"""
data_loader.py
----------------
Parses backend/data/all_data.txt, which contains THREE different
sections mixed in one text file:

1) A block of JSON-lines documents about driving-license procedures
   in Egypt (each line is a standalone JSON object).
2) A tab-separated table of driving schools / instructors
   (School Name, Phone, Area, Governorate).
3) A tab-separated table of car maintenance service pricing
   (engine_cc, service_type, interval_km, estimated_price_egp,
   service_center_type, city).

This module turns that raw file into clean Python objects that the
rest of the app (rag.py, main.py) can use.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "all_data.txt"


@dataclass
class LicenseDoc:
    id: str
    category: str
    topic: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        return f"[{self.category} - {self.topic}]\n{self.content}"


@dataclass
class School:
    name: str
    phone: str
    area: str
    governorate: str

    def as_text(self) -> str:
        return (
            f"مدرسة/مدرب سواقة: {self.name} - المنطقة: {self.area} - "
            f"المحافظة: {self.governorate} - رقم الهاتف: {self.phone}"
        )


@dataclass
class MaintenanceRow:
    engine_cc: str
    service_type: str
    interval_km: str
    estimated_price_egp: str
    service_center_type: str
    city: str

    def as_text(self) -> str:
        return (
            f"خدمة صيانة: {self.service_type} لسيارة سعة محرك {self.engine_cc} سي سي، "
            f"كل {self.interval_km} كم، السعر التقريبي {self.estimated_price_egp} جنيه، "
            f"في {self.service_center_type} بمدينة {self.city}."
        )


def _clean_line(line: str) -> str:
    return line.replace("\r", "").strip()


def load_raw_lines() -> List[str]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [_clean_line(l) for l in f.readlines()]


def parse_all_data():
    lines = load_raw_lines()

    license_docs: List[LicenseDoc] = []
    schools: List[School] = []
    maintenance: List[MaintenanceRow] = []

    section = "license"  # license -> schools -> maintenance
    schools_header_seen = False
    maintenance_header_seen = False

    for raw in lines:
        if raw == "":
            continue

        # --- Section 1: JSON-lines license docs ---
        if section == "license":
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                    license_docs.append(
                        LicenseDoc(
                            id=obj.get("id", ""),
                            category=obj.get("category", ""),
                            topic=obj.get("topic", ""),
                            content=obj.get("content", ""),
                            metadata=obj.get("metadata", {}),
                        )
                    )
                    continue
                except json.JSONDecodeError:
                    pass
            # First non-JSON line -> we've hit the schools table header
            section = "schools"

        # --- Section 2: schools table ---
        if section == "schools":
            if not schools_header_seen:
                schools_header_seen = True  # this line is the header, skip it
                continue
            if raw.lower().startswith("engine_cc"):
                # We actually reached the maintenance header without a blank line
                section = "maintenance"
                maintenance_header_seen = True
                continue
            parts = re.split(r"\t+", raw)
            if len(parts) >= 4:
                schools.append(
                    School(
                        name=parts[0].strip(),
                        phone=parts[1].strip(),
                        area=parts[2].strip(),
                        governorate=parts[3].strip(),
                    )
                )
                continue
            else:
                # Doesn't look like a school row anymore -> maintenance header
                section = "maintenance"
                continue

        # --- Section 3: maintenance table ---
        if section == "maintenance":
            if not maintenance_header_seen:
                maintenance_header_seen = True  # header row, skip it
                continue
            parts = re.split(r"\t+", raw)
            if len(parts) >= 6:
                maintenance.append(
                    MaintenanceRow(
                        engine_cc=parts[0].strip(),
                        service_type=parts[1].strip(),
                        interval_km=parts[2].strip(),
                        estimated_price_egp=parts[3].strip(),
                        service_center_type=parts[4].strip(),
                        city=parts[5].strip(),
                    )
                )

    return license_docs, schools, maintenance


if __name__ == "__main__":
    docs, schools, maint = parse_all_data()
    print(f"license_docs: {len(docs)}")
    print(f"schools: {len(schools)}")
    print(f"maintenance rows: {len(maint)}")
    print(docs[0].as_text()[:120] if docs else "no docs")
    print(schools[0].as_text() if schools else "no schools")
    print(maint[0].as_text() if maint else "no maintenance")
