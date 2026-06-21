"""tests/test_detection_init_exports.py — detection package public exports."""
import structverify.detection as detection


def test_detection_package_exports_public_api():
    for name in detection.__all__:
        assert hasattr(detection, name)

    assert detection.classify_domain is not None
    assert detection.detect_claims is not None
    assert detection.induce_schemas is not None
    assert detection.regenerate_schema is not None
