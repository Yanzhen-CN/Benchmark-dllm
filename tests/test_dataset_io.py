import json

from dllm_bench.datasets.ifeval import IFEvalSample, InstructionSpec
from dllm_bench.datasets.io import load_samples_file


def test_load_samples_file_builds_typed_ifeval_reference(tmp_path):
    path = tmp_path / "ifeval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "ifeval-1",
                    "prompt": "Use two bullets and mention Python.",
                    "reference": {
                        "form_constraints": [
                            {
                                "kind": "format:number_bullets",
                                "args": {"count": 2, "relation": "exactly"},
                            }
                        ],
                        "content_requirements": [
                            {
                                "kind": "keywords:existence",
                                "args": {"keywords": ["python"]},
                            }
                        ],
                        "target_length_words": 20,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = load_samples_file(path, "ifeval")[0]

    assert isinstance(sample.reference, IFEvalSample)
    assert sample.reference.form_constraints == [
        InstructionSpec(
            "format:number_bullets", {"count": 2, "relation": "exactly"}
        )
    ]
    assert sample.reference.content_requirements[0].kind == "keywords:existence"
    assert sample.reference.target_length_words == 20
