"""Tests for ollmatuning.cli — argument parsing and command dispatch."""
from unittest.mock import patch, MagicMock

from ollmatuning.cli import main, _resolve_runtime, _sort_candidates
from ollmatuning.recommend import Candidate


class TestResolveRuntime:
    def test_mlx_flag(self):
        args = MagicMock(mlx=True, gguf=False)
        assert _resolve_runtime(args) == "mlx"

    def test_gguf_flag(self):
        args = MagicMock(mlx=False, gguf=True)
        assert _resolve_runtime(args) == "ollama"

    def test_mlx_flag_overrides_apple_silicon(self):
        args = MagicMock(mlx=True, gguf=False)
        with patch("ollmatuning.cli.is_apple_silicon", return_value=True):
            assert _resolve_runtime(args) == "mlx"

    def test_default_apple_silicon_returns_mlx(self):
        args = MagicMock(mlx=False, gguf=False)
        with patch("ollmatuning.cli.is_apple_silicon", return_value=True):
            assert _resolve_runtime(args) == "mlx"

    def test_default_linux_returns_ollama(self):
        args = MagicMock(mlx=False, gguf=False)
        with patch("ollmatuning.cli.is_apple_silicon", return_value=False):
            assert _resolve_runtime(args) == "ollama"


class TestSortCandidates:
    def test_code_tools_first(self):
        cands = [
            Candidate("g", "fam", 7, 4000, ["general"]),
            Candidate("c", "fam", 7, 4000, ["code"]),
            Candidate("ct", "fam", 7, 4000, ["code", "tools"]),
            Candidate("t", "fam", 7, 4000, ["tools"]),
        ]
        sorted_ = _sort_candidates(cands)
        assert sorted_[0].categories == ["code", "tools"]
        assert sorted_[1].categories == ["code"]
        assert sorted_[2].categories == ["tools"]
        assert sorted_[3].categories == ["general"]

    def test_larger_first_within_tier(self):
        cands = [
            Candidate("small", "fam", 3, 2000, ["code"]),
            Candidate("large", "fam", 13, 8000, ["code"]),
        ]
        sorted_ = _sort_candidates(cands)
        assert sorted_[0].model == "large"


class TestMain:
    def test_detect_command_json(self, capsys):
        with patch("ollmatuning.cli.detect_system") as mock_detect:
            mock_info = MagicMock()
            mock_info.to_dict.return_value = {"os": "Linux", "ram_mb": 8192}
            mock_detect.return_value = mock_info
            with patch("ollmatuning.cli.is_apple_silicon", return_value=False):
                rc = main(["detect", "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Linux" in captured.out
        assert "8192" in captured.out

    def test_show_no_config(self):
        with patch("ollmatuning.cli.CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            rc = main(["show"])
        assert rc == 1

    def test_version_flag(self, capsys):
        try:
            main(["--version"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "ollmatuning" in captured.out

    def test_no_command_defaults_to_auto(self):
        with patch("ollmatuning.cli.cmd_auto") as mock_auto:
            mock_auto.return_value = 0
            with patch("ollmatuning.cli.detect_system"), \
                 patch("ollmatuning.cli.is_apple_silicon", return_value=False):
                main([])
            mock_auto.assert_called_once()
