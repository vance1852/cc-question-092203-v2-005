"""端到端回归：同一次 CLI 运行中图形数值与 JSON 逐项对应。

覆盖路径：基线评估、优化收敛、台数扫描、逐机平均、优化前后对比。
若任何一环重复乘除 1000，柱高/曲线与 results.json 的 *_gwh 字段就会
相差 1000 倍，下列断言立即失败。
"""

import json

import numpy as np
import pytest

from wind_farm_opt.cli import WindFarmOptimizerCLI
from wind_farm_opt.config import (
    EconomicConfig,
    OptimizationConfig,
    VisualizationConfig,
    WindFarmConfig,
)
from wind_farm_opt.visualization import plotting


@pytest.fixture
def small_config(tmp_path):
    return WindFarmConfig(
        n_turbines=8,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=6,
            max_iterations=3,
            min_spacing_multiple=5.0,
            seed=42,
        ),
        visualization=VisualizationConfig(
            save_dir=str(tmp_path / "out"),
            save_plots=True,
            show_plots=False,
            plot_wake_heatmap=False,
        ),
        economic=EconomicConfig(enable_analysis=False),
    )


@pytest.fixture
def captured(monkeypatch):
    captured = []
    monkeypatch.setattr(
        plotting.plt, "close", lambda fig=None: captured.append(fig)
    )
    return captured


def test_full_run_figures_match_json(small_config, captured, tmp_path):
    cli = WindFarmOptimizerCLI(small_config)
    cli._min_turbines = 4
    cli._max_turbines = 8
    cli.run_full_analysis(
        run_baseline=True,
        run_opt=True,
        run_econ=False,
        run_sweep=True,
        run_viz=True,
        save=True,
    )

    with open(tmp_path / "out" / "results.json", encoding="utf-8") as f:
        data = json.load(f)

    base_mwh = cli.baseline_result.net_aep
    opt_mwh = cli.optimized_result.net_aep
    base_gwh = base_mwh / 1e3
    opt_gwh = opt_mwh / 1e3

    # ---- JSON 与 FarmResult 一致（MWh -> GWh 只除一次）----
    assert data["baseline"]["net_aep_gwh"] == pytest.approx(base_gwh, rel=1e-12)
    assert data["optimized"]["net_aep_gwh"] == pytest.approx(opt_gwh, rel=1e-12)
    assert data["improvement"]["additional_aep_gwh"] == pytest.approx(
        (opt_mwh - base_mwh) / 1e3, rel=1e-12
    )
    # 合理物理量级：8 台 3.45MW 风机年发电应为几十 GWh，而非几万 GWh。
    assert 10.0 < base_gwh < 300.0
    assert 10.0 < opt_gwh < 300.0

    # ---- 优化前后对比图：柱高 == JSON 的 GWh 值 ----
    comp_fig = next(
        fig for fig in captured
        if fig._suptitle is not None and "关键指标对比" in fig._suptitle.get_text()
    )
    aep_ax, _, _, per_ax = comp_fig.axes
    bar_heights = [b.get_height() for b in aep_ax.containers[0]]
    assert bar_heights == pytest.approx(
        [data["baseline"]["net_aep_gwh"], data["optimized"]["net_aep_gwh"]],
        rel=1e-12,
    )
    per_heights = [b.get_height() for b in per_ax.containers[0]]
    assert per_heights == pytest.approx(
        [
            data["baseline"]["net_aep_gwh"] / 8,
            data["optimized"]["net_aep_gwh"] / 8,
        ],
        rel=1e-12,
    )

    # ---- 收敛曲线：基线横线 == JSON 基线 GWh；历史值与最优解同数量级 ----
    conv_fig = next(
        fig for fig in captured
        if any("优化收敛曲线" in ax.get_title() for ax in fig.axes)
    )
    ax = conv_fig.axes[0]
    best_line, _, baseline_line = ax.lines
    assert baseline_line.get_ydata()[0] == pytest.approx(base_gwh, rel=1e-9)
    # GA 在每代评估前记录历史，末代最优可能未入列；但历史值必须与
    # 最优解处于同一数量级（任何 1000 倍漂移都会越界）。
    history = np.asarray(best_line.get_ydata())
    assert np.all(history > 0.5 * base_gwh)
    assert np.all(history < 1.5 * opt_gwh)
    # 最优适应度（MWh）换算后与 JSON 优化净 AEP 一致。
    assert cli.optimize_result.best_fitness / 1e3 == pytest.approx(opt_gwh, rel=1e-6)

    # ---- 台数扫描：曲线 GWh == JSON aep_mwh / 1000 ----
    sweep_fig = next(
        fig for fig in captured
        if any("风机台数" in ax.get_title() for ax in fig.axes)
    )
    sweep_ax = sweep_fig.axes[0]
    expected_sweep_gwh = [v / 1e3 for v in data["turbine_sweep"]["aep_mwh"]]
    np.testing.assert_allclose(
        sweep_ax.lines[0].get_ydata(), expected_sweep_gwh, rtol=1e-12
    )
    np.testing.assert_array_equal(
        sweep_ax.lines[0].get_xdata(), data["turbine_sweep"]["n_turbines"]
    )

    # ---- PNG 与 JSON 文件都已落盘 ----
    for name in ("comparison.png", "convergence.png", "aep_vs_turbines.png",
                 "baseline_layout.png", "optimized_layout.png",
                 "baseline_losses.png", "optimized_losses.png"):
        path = tmp_path / "out" / name
        assert path.exists() and path.stat().st_size > 1000, f"{name} 未正常生成"


def test_sweep_restores_primary_farm_state(small_config, captured, tmp_path):
    """台数扫描后，主流程的风机/直径数组必须恢复为配置台数。"""
    cli = WindFarmOptimizerCLI(small_config)
    cli._min_turbines = 4
    cli._max_turbines = 6
    cli.run_full_analysis(
        run_baseline=True, run_opt=True, run_econ=False,
        run_sweep=True, run_viz=True, save=False,
    )
    assert len(cli.turbines) == small_config.n_turbines
    assert len(cli.rotor_diameters) == small_config.n_turbines
    assert cli.aep_calc is not None
    assert len(cli.aep_calc.turbines) == small_config.n_turbines
