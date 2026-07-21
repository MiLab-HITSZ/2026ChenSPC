from unittest.mock import patch

import spc_vlm_runtime as runtime


def llava_spec(models_root: str = "models") -> runtime.ModelSpec:
    return runtime.ModelSpec(
        name="LLaVA-1.6-34B-Instruct-transformers",
        backend="transformers",
        model="llava-v1.6-34b-hf",
        temperature=0.0,
        max_tokens=1,
        models_root=models_root,
        quantization="none",
        dtype="float16",
    )


def test_local_candidate_bundle_dispatches_llava_without_qwen_loading(tmp_path):
    runtime._MODEL_CACHE.clear()
    expected = (object(), object(), "llava_next")
    with patch("official_revis_adapter.load_model_bundle", return_value=expected) as load:
        assert runtime.local_candidate_bundle(llava_spec(str(tmp_path))) == expected
    load.assert_called_once_with("llava-v1.6-34b-hf")


def test_candidate_inputs_delegate_to_llava_adapter():
    expected = {"input_ids": object()}
    with patch("official_revis_adapter.build_inputs", return_value=expected) as build:
        actual = runtime.candidate_vlm_inputs(
            object(), object(), "llava_next", "question", None
        )
    assert actual is expected
    build.assert_called_once()
