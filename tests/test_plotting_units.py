"""绘图单位契约的回归测试。

所有用例都在保存 PNG 后拦截 ``plt.close`` 取回图形对象，直接检查
柱高/曲线坐标/注释文本，确保图上标注 GWh 的地方绘制的确实是 GWh 数值，
且与同一次运行的 FarmResult / JSON 数值逐项对应。
"""

import json

import numpy as np
import pytest

from wind_farm_opt.farm.aep import FarmResult, TurbineResult
from wind_farm_opt.optimization.ga import OptimizeResult
from wind_farm_opt.units import EnergyUnit
from wind_farm_opt.visualization import plotting


@pytest.fixture
def captured_figures(monkeypatch):
    """拦截绘图函数末尾的 plt.close，返回收集到的图形列表。"""
    captured = []
    monkeypatch.setattr(
        plotting.plt, "close", lambda fig=None: captured.append(fig)
    )
    return captured


def _make_farm_result(net_aep_mwh: float, n_turbines: int = 10,
                      wake_loss_pct: float = 8.0) -> FarmResult:
    gross_mwh = net_aep_mwh / (1.0 - wake_loss_pct / 100.0)
    per_turbine_net = net_aep_mwh / n_turbines
    per_turbine_gross = gross_mwh / n_turbines
    turbine_results = [
        TurbineResult(
            turbine_idx=i,
            name=f"T{i}",
            gross_aep=per_turbine_gross,
            net_aep=per_turbine_net,
            wake_loss=per_turbine_gross - per_turbine_net,
            wake_loss_pct=wake_loss_pct,
            capacity_factor=40.0,
            avg_effective_speed=8.0,
            dominant_wake_source=None,
        )
        for i in range(n_turbines)
    ]
    return FarmResult(
        gross_aep=gross_mwh,
        net_aep=net_aep_mwh,
        total_wake_loss=gross_mwh - net_aep_mwh,
        wake_loss_pct=wake_loss_pct,
        capacity_factor=40.0,
        total_installed_capacity=3.45 * n_turbines,
        turbine_results=turbine_results,
        sector_results={},
    )


def _find_figure(captured, keyword: str):
    for fig in captured:
        title = ""
        if fig._suptitle is not None:
            title = fig._suptitle.get_text()
        for ax in fig.axes:
            title += ax.get_title()
        if keyword in title:
            return fig
    raise AssertionError(f"未找到标题包含 {keyword!r} 的图形")


# ---------------------------------------------------------------------------
# plot_comparison：MWh 数值不得直接贴 GWh 标签
# ---------------------------------------------------------------------------

def test_plot_comparison_converts_mwh_to_gwh(tmp_path, captured_figures):
    baseline = _make_farm_result(12_345.6, n_turbines=10, wake_loss_pct=10.0)
    optimized = _make_farm_result(13_500.0, n_turbines=10, wake_loss_pct=7.5)

    plotting.plot_comparison(
        baseline, optimized, save_path=str(tmp_path / "comparison.png")
    )

    fig = _find_figure(captured_figures, "优化前后对比")
    aep_ax, loss_ax, cf_ax, per_ax = fig.axes

    # 净AEP 面板：柱高必须是 GWh（12.35 / 13.50），而不是原始 MWh 数值。
    bar_heights = [b.get_height() for b in aep_ax.containers[0]]
    assert bar_heights == pytest.approx([12.3456, 13.5], rel=1e-9)

    texts = " ".join(t.get_text() for t in aep_ax.texts)
    assert "12.35 GWh" in texts
    assert "13.50 GWh" in texts
    assert "12345" not in texts and "13500" not in texts
    assert aep_ax.get_ylabel() == "GWh"

    # 单机平均面板：12.3456 GWh / 10 台 = 1.23456 GWh/台。
    per_heights = [b.get_height() for b in per_ax.containers[0]]
    assert per_heights == pytest.approx([1.23456, 1.35], rel=1e-9)
    assert per_ax.get_ylabel() == "GWh/台"
    per_texts = " ".join(t.get_text() for t in per_ax.texts)
    assert "1.23 GWh/台" in per_texts

    # 百分比面板保持百分比，不参与能量换算。
    loss_heights = [b.get_height() for b in loss_ax.containers[0]]
    assert loss_heights == pytest.approx([10.0, 7.5])


def test_plot_comparison_improvement_uses_consistent_units(tmp_path, captured_figures):
    baseline = _make_farm_result(10_000.0, n_turbines=10, wake_loss_pct=10.0)
    optimized = _make_farm_result(11_000.0, n_turbines=10, wake_loss_pct=8.0)

    plotting.plot_comparison(
        baseline, optimized, save_path=str(tmp_path / "comparison2.png")
    )
    fig = _find_figure(captured_figures, "优化前后对比")
    aep_ax = fig.axes[0]
    # +10% 提升基于 GWh 与基于 MWh 的比值相同，但前提是两个柱使用同一单位。
    assert "提升 10.0%" in aep_ax.get_title()


# ---------------------------------------------------------------------------
# plot_convergence：显式单位入参，曲线与基线处于同一数量级
# ---------------------------------------------------------------------------

def _make_opt_result(best_fitness_mwh: float) -> OptimizeResult:
    history = [best_fitness_mwh - 200.0, best_fitness_mwh - 100.0, best_fitness_mwh]
    return OptimizeResult(
        best_positions=np.zeros((2, 2)),
        best_fitness=best_fitness_mwh,
        best_generation=2,
        convergence_history=history,
        mean_history=[v - 500.0 for v in history],
        final_population=np.zeros((1, 4)),
        final_fitness=np.array([best_fitness_mwh]),
    )


def test_plot_convergence_default_mwh(tmp_path, captured_figures):
    opt = _make_opt_result(50_000.0)
    plotting.plot_convergence(
        opt,
        baseline_aep=48_000.0,  # MWh
        save_path=str(tmp_path / "conv.png"),
    )

    fig = captured_figures[0]
    ax = fig.axes[0]

    best_line, mean_line, baseline_line = ax.lines
    np.testing.assert_allclose(
        best_line.get_ydata(), [49.8, 49.9, 50.0]
    )
    assert baseline_line.get_ydata()[0] == pytest.approx(48.0)

    assert ax.get_ylabel() == "净年发电量 (GWh)"
    legend = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "网格布局基线: 48.00 GWh" in legend

    info = ax.texts[0].get_text()
    assert "最优解: 50.00 GWh" in info
    # (50000-48000)/48000 = 4.1667%，使用同单位计算
    assert "相对提升: 4.17%" in info


def test_plot_convergence_kwh_input_normalizes(tmp_path, captured_figures):
    """quick_test.py 旧写法误传 kWh；现在显式声明 kWh 后图上仍为同一 GWh。"""
    opt = _make_opt_result(50_000.0)
    # 把同一组数值按 kWh 重新表达（×1000）并显式声明。
    opt_kwh = OptimizeResult(
        best_positions=opt.best_positions,
        best_fitness=opt.best_fitness * 1e3,
        best_generation=opt.best_generation,
        convergence_history=[v * 1e3 for v in opt.convergence_history],
        mean_history=[v * 1e3 for v in opt.mean_history],
        final_population=opt.final_population,
        final_fitness=opt.final_fitness * 1e3,
    )
    plotting.plot_convergence(
        opt_kwh,
        baseline_aep=48_000.0 * 1e3,
        aep_unit=EnergyUnit.KWH,
        save_path=str(tmp_path / "conv_kwh.png"),
    )
    ax = captured_figures[0].axes[0]
    best_line, _, baseline_line = ax.lines
    np.testing.assert_allclose(best_line.get_ydata(), [49.8, 49.9, 50.0])
    assert baseline_line.get_ydata()[0] == pytest.approx(48.0)


def test_plot_convergence_gwh_input(tmp_path, captured_figures):
    opt = _make_opt_result(50_000.0)
    opt_gwh = OptimizeResult(
        best_positions=opt.best_positions,
        best_fitness=50.0,
        best_generation=opt.best_generation,
        convergence_history=[49.8, 49.9, 50.0],
        mean_history=[49.3, 49.4, 49.5],
        final_population=opt.final_population,
        final_fitness=np.array([50.0]),
    )
    plotting.plot_convergence(
        opt_gwh, baseline_aep=48.0, aep_unit="GWh",
        save_path=str(tmp_path / "conv_gwh.png"),
    )
    ax = captured_figures[0].axes[0]
    best_line, _, baseline_line = ax.lines
    np.testing.assert_allclose(best_line.get_ydata(), [49.8, 49.9, 50.0])
    assert baseline_line.get_ydata()[0] == pytest.approx(48.0)


def test_plot_convergence_rejects_unknown_unit(tmp_path):
    opt = _make_opt_result(50_000.0)
    with pytest.raises(ValueError, match="未知能量单位"):
        plotting.plot_convergence(
            opt, baseline_aep=48_000.0, aep_unit="TWh",
            save_path=str(tmp_path / "bad.png"),
        )


# ---------------------------------------------------------------------------
# plot_aep_vs_turbines：台数扫描的 MWh 数组
# ---------------------------------------------------------------------------

def test_plot_aep_vs_turbines_mwh(tmp_path, captured_figures):
    n_list = [5, 10, 15]
    aep_mwh = [20_000.0, 38_000.0, 54_000.0]
    lcoe = [0.30, 0.28, 0.27]

    plotting.plot_aep_vs_turbines(
        n_list, aep_mwh, lcoe_list=lcoe,
        save_path=str(tmp_path / "sweep.png"),
    )

    fig = captured_figures[0]
    ax1, ax2 = fig.axes
    np.testing.assert_allclose(ax1.lines[0].get_ydata(), [20.0, 38.0, 54.0])
    assert ax1.get_ylabel() == "净年发电量 (GWh)"
    annotations = " ".join(t.get_text() for t in ax1.texts)
    assert "20.0" in annotations and "54.0" in annotations
    # LCOE 面板不受能量换算影响。
    np.testing.assert_allclose(ax2.lines[0].get_ydata(), lcoe)


def test_plot_aep_vs_turbines_unit_overloads_match(tmp_path, captured_figures):
    n_list = [5, 10]
    aep_mwh = [20_000.0, 38_000.0]

    plotting.plot_aep_vs_turbines(
        n_list, [v * 1e3 for v in aep_mwh], aep_unit="kWh",
        save_path=str(tmp_path / "sweep_kwh.png"),
    )
    ax_kwh = captured_figures[0].axes[0]
    np.testing.assert_allclose(ax_kwh.lines[0].get_ydata(), [20.0, 38.0])

    plotting.plot_aep_vs_turbines(
        n_list, [20.0, 38.0], aep_unit=EnergyUnit.GWH,
        save_path=str(tmp_path / "sweep_gwh.png"),
    )
    ax_gwh = captured_figures[1].axes[0]
    np.testing.assert_allclose(ax_gwh.lines[0].get_ydata(), [20.0, 38.0])
