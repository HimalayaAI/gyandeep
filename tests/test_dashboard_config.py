import pytest

from dashboard.backend import config


def test_validate_config_ok():
    config.validate_config()


def test_validate_config_missing_dirs(monkeypatch, tmp_path):
    missing_templates = tmp_path / "no-templates"
    missing_static = tmp_path / "no-static"

    monkeypatch.setattr(config, "TEMPLATES_DIR", str(missing_templates))
    monkeypatch.setattr(config, "STATIC_DIR", str(missing_static))

    with pytest.raises(RuntimeError):
        config.validate_config()
