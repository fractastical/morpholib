import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "morphoeval"))

import problem_fixer as pf  # noqa: E402


def run(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE, text=True)


def test_queue_matches_leaderboard_and_selection() -> None:
    queue = pf.load_queue()
    assert len(queue["tasks"]) == 12
    assert [task["id"] for task in queue["tasks"]] == [f"fix-{n:03d}" for n in range(1, 13)]
    assert pf.parse_selection("1-3,12") == [1, 2, 3, 12]
    assert pf.parse_selection("fix-004") == [4]


def test_prompt_contract_and_allowlist() -> None:
    task = pf.tasks_by_number()[3]
    prompt = pf.build_prompt(task)
    assert "Problem-Fixer fix-003" in prompt
    assert "Do **not** merge your own PR" in prompt
    assert "external_publication_performed" in prompt
    assert pf.path_is_allowed(task, "morphoeval/problem_scouts/verify_rule30_delphi_sync.py")
    assert pf.path_is_allowed(task, "morphoeval/problem_fixers/results/fix-003.json")
    assert not pf.path_is_allowed(task, "predictions/prediction_registry.json")
    assert not pf.path_is_allowed(task, ".github/workflows/problem-scout-100.yml")
    assert not pf.path_is_allowed(task, "morphoeval/problem_fixers/fix_queue.json")


def test_guard_accepts_only_allowlisted_reported_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "morphoeval/problem_scouts").mkdir(parents=True)
    (repo / "morphoeval/problem_fixers/results").mkdir(parents=True)
    (repo / "morphoeval/problem_fixers/reports").mkdir(parents=True)
    run(repo, "init")
    run(repo, "config", "user.email", "fixer@example.test")
    run(repo, "config", "user.name", "Fixer Test")
    (repo / "README.md").write_text("base\n")
    run(repo, "add", ".")
    run(repo, "commit", "-m", "base")

    changed = "morphoeval/problem_scouts/verify_rule30_delphi_sync.py"
    (repo / changed).write_text("print('ok')\n")
    result = {
        "fix_id": "fix-003",
        "candidate_rank": 3,
        "candidate_id": "scout-010-c",
        "status": "fixed",
        "summary": "test",
        "root_cause": "test",
        "changed_files": [changed],
        "evidence": ["test"],
        "validation": [],
        "blockers": [],
        "external_publication_performed": False,
    }
    (repo / "morphoeval/problem_fixers/results/fix-003.json").write_text(json.dumps(result))
    (repo / "morphoeval/problem_fixers/reports/fix-003.md").write_text("# report\n")
    run(repo, "add", ".")
    run(repo, "commit", "-m", "fix")

    task = pf.tasks_by_number()[3]
    report = pf.guard_task(task, repo, "HEAD~1", run_validator_checks=False)
    assert report["passed"], report
    assert report["auto_merge_eligible"]

    (repo / "predictions").mkdir()
    (repo / "predictions/prediction_registry.json").write_text("{}\n")
    run(repo, "add", ".")
    run(repo, "commit", "-m", "forbidden")
    report = pf.guard_task(task, repo, "HEAD~2", run_validator_checks=False)
    assert not report["passed"]
    assert any("outside the allowlist" in error for error in report["errors"])
