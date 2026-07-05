"""New AI-brain scheduler jobs are registered and config-gated."""
import scheduler.job_scheduler as js


def test_tournament_jobs_registered():
    s = js.create_scheduler()
    ids = {j.id for j in s.get_jobs()}
    assert "weekly_tournament" in ids
    assert "missed_opportunity_scan" in ids


def test_weekly_tournament_respects_disabled_flag(monkeypatch):
    monkeypatch.setitem(js._cfg["strategy_tournament"], "enabled", False)
    js.job_weekly_tournament()  # early-returns; must not raise or touch network


def test_missed_opp_scan_respects_disabled_flag(monkeypatch):
    monkeypatch.setitem(
        js._cfg["strategy_tournament"]["missed_opportunity"], "enabled", False
    )
    js.job_missed_opportunity_scan()  # early-returns; must not raise or touch network
