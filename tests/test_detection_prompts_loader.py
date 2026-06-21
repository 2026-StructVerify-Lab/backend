"""tests/test_detection_prompts_loader.py — prompts_loader unit tests (API 키 불필요)."""
import yaml

from structverify.detection.prompts_loader import (
    few_shot_examples_from_pack,
    format_few_shot_block,
    inject_few_shot,
    load_domain_pack,
    load_domain_prompts,
    prompts_yaml_path,
)


def test_load_domain_pack_missing_returns_none():
    assert load_domain_pack("nonexistent_domain_xyz", {"domain_packs_dir": "/no/such/dir"}) is None


def test_load_domain_pack_reads_yaml(tmp_path):
    pack_dir = tmp_path / "economy"
    pack_dir.mkdir()
    data = {"domain": "economy", "few_shot_examples": ["예시 1"]}
    (pack_dir / "prompts.yaml").write_text(
        yaml.dump(data, allow_unicode=True), encoding="utf-8"
    )
    config = {"domain_packs_dir": str(tmp_path)}
    assert prompts_yaml_path("economy", config) == str(pack_dir / "prompts.yaml")
    loaded = load_domain_pack("economy", config)
    assert loaded == data
    assert load_domain_prompts("economy", config) == data


def test_few_shot_helpers_no_op_when_empty():
    prompt = "base prompt"
    assert inject_few_shot(prompt, None) == prompt
    assert format_few_shot_block([]) == ""
    assert few_shot_examples_from_pack({"few_shot_examples": []}) == []


def test_inject_few_shot_appends_block():
    pack = {
        "few_shot_examples": [
            {"input": "출생아 2만명", "output": "true"},
            "plain string example",
        ],
    }
    out = inject_few_shot("BASE", pack)
    assert out.startswith("BASE")
    assert "[도메인 few-shot 예시]" in out
    assert "출생아 2만명" in out
    assert "plain string example" in out
