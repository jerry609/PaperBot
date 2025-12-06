import math
from influence.calculator import InfluenceCalculator
from scholar_tracking.models.influence import InfluenceResult


class DummyPaper:
    def __init__(self, year=None, citation_count=10, venue="CCS"):
        self.year = year
        self.citation_count = citation_count
        self.venue = venue
        self.title = "Dummy"
        self.paper_id = "p1"
        self.authors = []
        self.abstract = ""
        self.tldr = ""
        self.github_url = None
        self.has_code = False


def test_recency_factor_newer_higher():
    calc = InfluenceCalculator({"weights": {"recency_half_life_years": 5}})
    recent = DummyPaper(year=2024)
    old = DummyPaper(year=2010)
    r_recent: InfluenceResult = calc.calculate(recent, None)
    r_old: InfluenceResult = calc.calculate(old, None)
    assert r_recent.total_score >= r_old.total_score


def test_recency_factor_bounds():
    calc = InfluenceCalculator({"weights": {"recency_half_life_years": 5}})
    paper = DummyPaper(year=2000)
    res = calc.calculate(paper, None)
    # recency factor 下限 0.3
    assert res.metrics_breakdown["weights"]["recency_factor"] >= 0.3

