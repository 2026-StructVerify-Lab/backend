def fetch_kosis_data(org_id: str, tbl_id: str, time_reference: str):
    year_match = re.search(r'(\d{4})', time_reference)
    year = year_match.group(1) if year_match else "2024"

    period_strategies = [
        {"prdSe": "Y",  "startPrdDe": year,         "endPrdDe": year},
        {"prdSe": "M",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}12"},
        {"prdSe": "Q",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}04"},
    ]

    for strategy in period_strategies:
        base_params = {
            "method": "getList", "apiKey": KOSIS_API_KEY,
            "format": "json", "jsonVD": "Y",
            "orgId": org_id, "tblId": tbl_id,
            "itmId": "ALL", "objL1": "ALL", "objL2": "ALL",
            "prdSe": strategy["prdSe"],
            "startPrdDe": strategy["startPrdDe"],
            "endPrdDe": strategy["endPrdDe"],
        }
        data = _try_with_objL_escalation(base_params)
        # ← 임시 디버그
        if tbl_id == "DT_1DE9058S":
            print(f"  [DEBUG] prdSe={strategy['prdSe']} → {type(data).__name__} {str(data)[:80]}")
        if data is not None:
            return data