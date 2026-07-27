"""M8-16: 适配器推理侧集成测试 — shade 切换通知宿主/输出加载指令。"""

from __future__ import annotations

import json
from pathlib import Path

from omnimem.internalize.lora_train import LoRATrainer


class TestInferenceDirective:
    def test_switch_without_adapter_emits_unload(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        result = t.switch_shade("work")
        assert result["status"] == "switched"
        assert result["inference"]["action"] == "unload_adapter"
        t.close()

    def test_switch_with_ready_adapter_emits_load(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        adapter_dir = omni_tmp_path / "ext_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        reg = t.register_external_adapter(adapter_dir, shade="work")
        assert reg["status"] == "registered"
        result = t.switch_shade("work")
        directive = result["inference"]
        assert directive["action"] == "load_adapter"
        assert directive["adapter_path"] == str(adapter_dir)
        assert directive["adapter_id"] == reg["adapter_id"]
        t.close()

    def test_get_inference_directive_reflects_active_shade(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        t.switch_shade("social")
        d = t.get_inference_directive()
        assert d["shade"] == "social"
        t.close()


class TestHostNotification:
    def test_hook_receives_directive_on_switch(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        received = []
        t.register_shade_change_hook(received.append)
        t.switch_shade("learning")
        assert len(received) == 1
        assert received[0]["shade"] == "learning"
        t.close()

    def test_hook_exception_does_not_break_switch(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        def bad_hook(_d):
            raise RuntimeError("host gone")
        t.register_shade_change_hook(bad_hook)
        result = t.switch_shade("work")
        assert result["status"] == "switched"
        t.close()

    def test_active_adapter_json_written(self, omni_tmp_path: Path):
        t = LoRATrainer(data_dir=omni_tmp_path)
        t.switch_shade("dark")
        marker = omni_tmp_path / "active_adapter.json"
        assert marker.exists()
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data["shade"] == "dark"
        assert data["action"] in ("load_adapter", "unload_adapter")
        t.close()
