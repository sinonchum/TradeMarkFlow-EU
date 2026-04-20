"""
TradeMarkFlow-EU — 端到端清查测试

完整流程：
1. 准备样本商标数据（含真实嵌入向量）
2. 批量写入 LanceDB
3. 执行清查检索（SQL 过滤 + 向量相似度 + 三维分析）
4. 输出风险评估报告
"""

import json
import sys
import os
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import TrademarkRecord, init_trademark_table, DB_PATH, TABLE_NAME, upsert_trademarks
from app.services.embeddings import embed_trademark_name, embed_query, similarity


# ═══════════════════════════════════════════════════════════════
#  1. 准备样本数据（模拟 EUIPO 已注册商标）
# ═══════════════════════════════════════════════════════════════

SAMPLE_MARKS = [
    {
        "application_number": "018912345",
        "mark_name": "SolarFlow",
        "nice_classes": [9, 37],
        "goods_services": ["Solar panels", "Photovoltaic cells", "Installation of solar energy systems"],
        "applicant_name": "SunTech GmbH",
        "applicant_country": "DE",
        "filing_date": "2024-03-15",
        "registration_date": "2024-09-15",
        "status": "registered",
    },
    {
        "application_number": "017654321",
        "mark_name": "SolFlux",
        "nice_classes": [9, 11],
        "goods_services": ["Solar water heaters", "Solar collectors"],
        "applicant_name": "Energie Verte SAS",
        "applicant_country": "FR",
        "filing_date": "2023-06-20",
        "registration_date": "2024-01-10",
        "status": "registered",
    },
    {
        "application_number": "016987654",
        "mark_name": "SolarFlex",
        "nice_classes": [35, 37],
        "goods_services": ["Retail of solar equipment", "Construction consulting"],
        "applicant_name": "GreenBuild BV",
        "applicant_country": "NL",
        "filing_date": "2022-11-05",
        "status": "registered",
    },
    {
        "application_number": "015432198",
        "mark_name": "SunnyEnergy",
        "nice_classes": [9, 42],
        "goods_services": ["Energy management software", "SaaS for energy monitoring"],
        "applicant_name": "BrightWatt Ltd",
        "applicant_country": "IE",
        "filing_date": "2023-01-20",
        "status": "registered",
    },
    {
        "application_number": "014567890",
        "mark_name": "VoltPower",
        "nice_classes": [9],
        "goods_services": ["Batteries", "Rechargeable batteries", "Power banks"],
        "applicant_name": "ElectraCorp SpA",
        "applicant_country": "IT",
        "filing_date": "2022-08-10",
        "status": "registered",
    },
    {
        "application_number": "013456789",
        "mark_name": "EcoVolt",
        "nice_classes": [9, 37, 42],
        "goods_services": ["Electric vehicle charging stations", "Installation services", "Energy consulting"],
        "applicant_name": "ChargePoint Europe AB",
        "applicant_country": "SE",
        "filing_date": "2024-01-30",
        "status": "application",
    },
    {
        "application_number": "012345678",
        "mark_name": "FluxEnergy",
        "nice_classes": [4, 37],
        "goods_services": ["Electricity supply", "Maintenance of power plants"],
        "applicant_name": "Nucléaire Moderne SA",
        "applicant_country": "FR",
        "filing_date": "2023-09-15",
        "registration_date": "2024-04-01",
        "status": "registered",
    },
    {
        "application_number": "011234567",
        "mark_name": "PhotoSol",
        "nice_classes": [9],
        "goods_services": ["Photovoltaic modules", "Solar inverters"],
        "applicant_name": "Deutsche Solar AG",
        "applicant_country": "DE",
        "filing_date": "2021-12-01",
        "registration_date": "2022-06-01",
        "status": "registered",
    },
    {
        "application_number": "010987123",
        "mark_name": "ZephyrWind",
        "nice_classes": [7, 37],
        "goods_services": ["Wind turbines", "Installation of wind energy systems"],
        "applicant_name": "NordicWind ApS",
        "applicant_country": "DK",
        "filing_date": "2024-02-10",
        "status": "published",
    },
    {
        "application_number": "009876543",
        "mark_name": "AtomPower",
        "nice_classes": [4, 37, 42],
        "goods_services": ["Nuclear energy generation", "Power plant construction", "Energy engineering"],
        "applicant_name": "EuroAtom SAS",
        "applicant_country": "FR",
        "filing_date": "2023-07-01",
        "registration_date": "2024-02-15",
        "status": "registered",
    },
    {
        "application_number": "008765432",
        "mark_name": "TerraGreen",
        "nice_classes": [31, 35],
        "goods_services": ["Organic produce", "Retail of organic food"],
        "applicant_name": "BioMarket SRL",
        "applicant_country": "IT",
        "filing_date": "2023-04-20",
        "status": "registered",
    },
    {
        "application_number": "007654321",
        "mark_name": "SoleilFlux",
        "nice_classes": [9, 37],
        "goods_services": ["Panneaux solaires", "Installation de systèmes solaires"],
        "applicant_name": "France Solaire EURL",
        "applicant_country": "FR",
        "filing_date": "2024-05-10",
        "status": "application",
    },
]


def populate_lancedb():
    """将样本数据（含真实嵌入向量）写入 LanceDB。"""
    print("=" * 60)
    print("Step 1: 生成嵌入向量 + 写入 LanceDB")
    print("=" * 60)

    records = []
    for sample in SAMPLE_MARKS:
        # 生成商标名称的嵌入向量
        mark_name = sample["mark_name"]
        nice_classes = sample.get("nice_classes", [])
        text_vec = embed_trademark_name(mark_name, nice_classes)

        record = TrademarkRecord(
            id=str(uuid.uuid4()),
            application_number=sample["application_number"],
            mark_name=mark_name,
            mark_type="word",
            nice_classes=nice_classes,
            goods_services=sample.get("goods_services", []),
            applicant_name=sample.get("applicant_name", ""),
            applicant_country=sample.get("applicant_country", ""),
            filing_date=sample.get("filing_date"),
            registration_date=sample.get("registration_date"),
            status=sample.get("status", "registered"),
            jurisdiction="EU",
            source="euipo",
            text_vector=text_vec,
            image_vector=[0.0] * 768,  # 固定 768 维（Schema 要求），非图形商标填零
            raw_url=f"https://euipo.europa.eu/eSearch/#details/trademarks/{sample['application_number']}",
            scraped_at=datetime.utcnow().isoformat(),
        )
        records.append(record)
        print(f"  ✓ {mark_name:25s} → {len(text_vec)}d vector | Nice {nice_classes}")

    # 删除占位行，写入真实数据
    import lancedb
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    # 清除占位行
    table.delete("id = '__placeholder__'")

    # 批量写入 — 使用 add() 直接追加（新表场景）
    dicts = [r.model_dump() for r in records]
    table.add(dicts)

    print(f"\n✅ 已写入 {len(records)} 条商标（含 {len(text_vec)} 维嵌入向量）")
    print(f"   表总行数: {table.count_rows()}")
    return table


# ═══════════════════════════════════════════════════════════════
#  2. 执行清查检索
# ═══════════════════════════════════════════════════════════════

def run_clearance_search(
    mark_name: str,
    nice_classes: list[int],
    business_description: str = "",
):
    """
    完整的清查检索流程：
    1. 生成查询向量
    2. LanceDB 混合检索（向量 + SQL 过滤）
    3. 三维相似度分析（音似/形似/义似）
    4. 绝对理由风险评估
    """
    print("\n" + "=" * 60)
    print(f"Step 2: 清查检索 — '{mark_name}' (Nice {nice_classes})")
    print("=" * 60)

    import lancedb
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    # 生成查询向量
    query_vec = embed_query(mark_name)

    # ── 向量检索（语义相似） ──
    print("\n[向量检索] Top 10 语义相似商标:")
    vec_results = (
        table.search(query_vec, vector_column_name="text_vector")
        .metric("cosine")
        .limit(10)
        .to_list()
    )

    for i, r in enumerate(vec_results):
        score = 1.0 - r.get("_distance", 1.0)  # cosine distance → similarity
        r["similarity_score"] = round(score, 4)
        print(f"  {i+1:2d}. {r['mark_name']:25s}  sim={score:.4f}  Nice={r.get('nice_classes', [])}  "
              f"| {r.get('applicant_country', '?')} {r.get('applicant_name', '')}")

    # ── SQL 过滤（尼斯分类重叠） ──
    print(f"\n[SQL 过滤] 同时包含 Nice {nice_classes} 的商标:")
    for nc in nice_classes:
        sql_results = (
            table.search()
            .where(f"array_contains(nice_classes, {nc})")
            .where("status = 'registered'")
            .to_list()
        )
        names = [r["mark_name"] for r in sql_results]
        print(f"  Class {nc}: {', '.join(names) if names else '(无)'}")

    # ── 三维分析 ──
    print("\n[三维相似度分析]")

    # Phonetic engine — 跨语言音似分析
    TRANSLITERATION_MAP = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ç": "c", "ñ": "n",
    }

    def phonetic_similarity(a: str, b: str) -> float:
        na = a.lower().strip()
        nb = b.lower().strip()
        for orig, repl in TRANSLITERATION_MAP.items():
            na = na.replace(orig, repl)
            nb = nb.replace(orig, repl)
        na = "".join(c for c in na if c.isalnum())
        nb = "".join(c for c in nb if c.isalnum())
        if na == nb:
            return 1.0
        max_len = max(len(na), len(nb))
        if max_len == 0:
            return 1.0
        common = sum(1 for x, y in zip(na, nb) if x == y)
        return common / max_len

    print(f"  {'商标':25s} {'语义':>8s} {'音似':>8s} {'综合':>8s}  风险")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")

    risks = []
    for r in vec_results:
        if r["mark_name"] == mark_name:
            continue

        semantic = r["similarity_score"]

        # 音似分析
        phonetic = phonetic_similarity(mark_name, r["mark_name"])

        # 综合分数 (语义 0.5 + 音似 0.5 for word marks)
        overall = 0.5 * semantic + 0.5 * phonetic

        if overall > 0.80:
            risk = "🔴 HIGH"
        elif overall > 0.65:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"

        risks.append({
            "mark_name": r["mark_name"],
            "semantic": round(semantic, 4),
            "phonetic": round(phonetic, 4),
            "overall": round(overall, 4),
            "risk": risk,
        })

        print(f"  {r['mark_name']:25s} {semantic:>8.4f} {phonetic:>8.4f} {overall:>8.4f}  {risk}")

    # ── 绝对理由评估 ──
    print("\n[绝对理由风险评估 — EUTMR Art. 7]")
    grounds = []

    mark_lower = mark_name.lower()

    # Art. 7(1)(c) 描述性检查
    descriptive_terms = {"solar", "eco", "green", "smart", "digital", "cloud", "fast"}
    for term in descriptive_terms:
        if term in mark_lower:
            grounds.append(f"Art. 7(1)(c) — 包含描述性词 '{term}'，可能缺乏显著性")
            break

    # Art. 7(1)(b) 极短标记
    if len(mark_name) <= 3:
        grounds.append("Art. 7(1)(b) — 标记极短，可能缺乏显著性")

    if grounds:
        for g in grounds:
            print(f"  ⚠️  {g}")
    else:
        print("  ✅ 未发现明显的绝对理由驳回风险")

    # ── 总结 ──
    high_risks = [r for r in risks if "HIGH" in r["risk"]]
    print(f"\n{'='*60}")
    print(f"📋 清查报告: '{mark_name}'")
    print(f"   推荐尼斯分类: {nice_classes}")
    print(f"   检索到 {len(risks)} 个潜在相似商标")
    print(f"   高风险: {len(high_risks)} 个")
    if high_risks:
        print(f"   最高风险: {high_risks[0]['mark_name']} (overall={high_risks[0]['overall']:.4f})")
        print(f"   ⚠️  建议：提交前需人工复核，评估混淆可能性 (Art. 8(1)(b) EUTMR)")
    else:
        print(f"   ✅ 整体风险: LOW — 可推进注册流程")
    print(f"{'='*60}")

    return risks, grounds


# ═══════════════════════════════════════════════════════════════
#  3. 执行
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 清空旧数据，重新初始化
    import shutil
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DB_PATH)
    lancedb_dir = os.path.join(db_path, "eu_trademarks.lance")
    if os.path.exists(lancedb_dir):
        shutil.rmtree(lancedb_dir)

    # 初始化表
    init_trademark_table(db_path)

    # 填充数据
    populate_lancedb()

    # 测试 1: 查询 "SolarFlux"（与 SolarFlow/SolFlux/SolarFlex 高度相似）
    print("\n" + "🔍" * 30)
    run_clearance_search("SolarFlux", [9, 37], "Solar panel manufacturing and installation")

    # 测试 2: 查询 "AtomPower"（与现有 AtomPower 精确冲突）
    print("\n\n" + "🔍" * 30)
    run_clearance_search("AtomPower", [4, 37], "Nuclear energy generation")
