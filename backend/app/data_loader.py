"""
data_loader.py
----------------
Parses backend/data/all_data.txt, which contains THREE different
sections mixed in one text file:

1) A block of JSON-lines documents about driving-license procedures,
   policies, and traffic-unit directories in Egypt (each line is a
   standalone JSON object).
2) A tab-separated table of driving schools / instructors
   (School Name, Area, Governorate). A Phone column is supported if
   present in the data, but the current dataset does not include phone
   numbers for any school, so `phone` defaults to "" when missing.
3) A tab-separated table of car maintenance service pricing
   (engine_cc, service_type, interval_km, estimated_price_egp,
   service_center_type, city).

Section boundaries are detected by the CONTENT of header lines
("School Name" / "Governorate" starts the schools table, "engine_cc"
starts the maintenance table) rather than by guessing from column
counts. Column-count guessing broke silently in the past whenever a
row didn't have the exact expected number of fields (e.g. the schools
table having no phone data at all).
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
    area: str
    governorate: str
    phone: str = ""  # not available for most/any rows in the current data

    def as_text(self) -> str:
        base = f"مدرسة/مدرب سواقة: {self.name} - المنطقة: {self.area} - المحافظة: {self.governorate}"
        if self.phone:
            base += f" - رقم الهاتف: {self.phone}"
        return base


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


def _is_schools_header(raw: str) -> bool:
    low = raw.lower()
    return "school name" in low or ("governorate" in low and "\t" in raw)


def _is_maintenance_header(raw: str) -> bool:
    return raw.lower().startswith("engine_cc")


def parse_all_data():
    lines = load_raw_lines()

    license_docs: List[LicenseDoc] = []
    schools: List[School] = []
    maintenance: List[MaintenanceRow] = []

    section = "license"  # license -> schools -> maintenance

    for raw in lines:
        if raw == "":
            continue

        # --- Section 1: JSON-lines license/policy/directory docs ---
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
                except json.JSONDecodeError:
                    pass  # malformed json line -> skip, stay in license section
                continue

            if _is_schools_header(raw):
                section = "schools"
            # any other stray non-JSON line while still in "license" is
            # ignored rather than mis-triggering a section change
            continue

        # --- Section 2: schools table ---
        if section == "schools":
            if _is_maintenance_header(raw):
                section = "maintenance"
                continue

            parts = [p.strip() for p in re.split(r"\t+", raw) if p.strip() != ""]
            if len(parts) >= 4:
                # name, phone, area, governorate
                schools.append(
                    School(name=parts[0], phone=parts[1], area=parts[2], governorate=parts[3])
                )
            elif len(parts) == 3:
                # name, area, governorate (no phone column in the data)
                schools.append(School(name=parts[0], area=parts[1], governorate=parts[2]))
            # rows with < 3 usable fields are malformed -> skipped, but we
            # STAY in the schools section (no more guessing our way into
            # "maintenance" based on a single bad row)
            continue

        # --- Section 3: maintenance table ---
        if section == "maintenance":
            parts = [p.strip() for p in re.split(r"\t+", raw) if p.strip() != ""]
            if len(parts) >= 6:
                maintenance.append(
                    MaintenanceRow(
                        engine_cc=parts[0],
                        service_type=parts[1],
                        interval_km=parts[2],
                        estimated_price_egp=parts[3],
                        service_center_type=parts[4],
                        city=parts[5],
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