from main import build_parser


def test_cli_track_flags():
    parser = build_parser()
    args = parser.parse_args(["track", "--mode", "academic", "--repro", "--report-template", "academic_report.md.j2"])
    assert args.command == "track"
    assert args.mode == "academic"
    assert args.repro is True
    assert args.report_template == "academic_report.md.j2"


def test_cli_render_latest_meta():
    parser = build_parser()
    args = parser.parse_args(["render-report"])
    assert args.command == "render-report"
    assert args.meta is None

