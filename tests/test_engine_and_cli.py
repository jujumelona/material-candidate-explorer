from __future__ import annotations

import json
from pathlib import Path

from discovery_os.cli import build_demo_engine, main
from discovery_os.schemas import (
    CandidateValidationStatus,
    ClaimLevel,
    DiscoveryDomain,
    StopReason,
)


def test_mock_model_runs_the_engine_end_to_end(tmp_path) -> None:
    output = tmp_path / "engine-run"
    engine = build_demo_engine(output)

    report = engine.run(
        "Screen a general material fixture without making a discovery claim.",
        max_cycles=1,
        domain_hint=DiscoveryDomain.GENERAL_MATERIALS,
    )

    assert report.run_id.startswith("RUN-")
    assert report.goal.domain == DiscoveryDomain.GENERAL_MATERIALS
    assert report.stop_decision.reason_code == StopReason.MAX_CYCLES_REACHED
    assert len(report.candidate_reports) == 2
    assert all(
        item.validation.status != CandidateValidationStatus.EXPERIMENTALLY_VALIDATED
        for item in report.candidate_reports
    )
    assert all(
        item.validation.claim_level
        in {ClaimLevel.GENERATED, ClaimLevel.COMPUTATIONALLY_PLAUSIBLE}
        for item in report.candidate_reports
    )
    assert engine.store.checkpoint is not None
    assert engine.store.checkpoint.state.cycle == 1
    assert engine.store.checkpoint.evidence
    assert (output / report.run_id / "checkpoint.json").is_file()
    report_files = list(
        (output / report.run_id / "artifacts" / "reports").glob(
            f"{report.run_id}-*.json"
        )
    )
    assert len(report_files) == 1


def test_cli_demo_emits_machine_readable_summary_and_report(tmp_path, capsys) -> None:
    output = tmp_path / "cli-run"

    exit_code = main(
        [
            "demo",
            "--goal",
            "Screen deterministic composition fixtures.",
            "--domain",
            DiscoveryDomain.GENERAL_MATERIALS,
            "--max-cycles",
            "1",
            "--output",
            str(output),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["domain"] == DiscoveryDomain.GENERAL_MATERIALS
    assert summary["candidates"] == 2
    assert summary["stop_reason"] == StopReason.MAX_CYCLES_REACHED
    assert "no discovery claim" in summary["notice"]
    report_path = Path(summary["report"])
    assert report_path.is_file()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["run_id"] == summary["run_id"]
