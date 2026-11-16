"""Exercise local checkpoint loading and real CLI reporting without downloads."""

import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.hf_baseline import main as hf_main
from benchmarks.manual_generate import main as manual_main
from benchmarks.verify import main as verify_main


def test_saved_qwen_checkpoint_through_all_clis(tiny_backend, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoint = tmp_path / "tiny qwen"
    tiny_backend.model.save_pretrained(checkpoint)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"word{i}": i for i in range(4, 97)})
    tokenizer_impl = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        chat_template="{{ bos_token }} {% for m in messages %}{{ m['content'] }} {% endfor %}",
    )
    tokenizer.save_pretrained(checkpoint)
    common = [
        "--model", str(checkpoint), "--local-files-only",
        "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "3",
        "--prompt", "word4 word5", "--prompt", "word6 word7 word8 word9",
    ]
    reports = {}
    for name, entrypoint in (("verify", verify_main), ("hf", hf_main), ("manual", manual_main)):
        path = tmp_path / f"{name}.json"
        extra = [] if name == "verify" else ["--warmup", "1", "--repetitions", "1"]
        assert entrypoint([*common, *extra, "--output", str(path)]) == 0
        reports[name] = json.loads(path.read_text())

    assert reports["verify"]["passed"] is True
    assert reports["verify"]["comparisons"] == 4
    assert reports["verify"]["environment"]["device"] == "cpu"
    assert reports["verify"]["environment"]["dtype"] == "torch.float32"
    assert reports["verify"]["environment"]["gpu"] is None
    assert reports["verify"]["environment"]["chat_template_sha256"] == reports["verify"]["generation"]["chat_template_sha256"]
    for hf_record, manual_record in zip(reports["hf"]["records"], reports["manual"]["records"], strict=True):
        assert hf_record["input_ids"] == manual_record["input_ids"]
        assert hf_record["output_ids"] == manual_record["output_ids"]
        assert hf_record["generation_seconds"] > 0
        assert manual_record["generation_seconds"] > 0
        assert hf_record["cuda_peak_allocated_bytes"] is None

