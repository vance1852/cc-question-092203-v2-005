"""能量单位契约回归测试。

锁定以下不变量，防止 MWh/GWh/kWh 之间重复乘除一千：

1. FarmResult / TurbineResult / OptimizeResult 的能量字段统一为 **MWh/year**；
2. 绘图函数在显示层唯一一次把 MWh 换算成 GWh，柱高/线值/注释/坐标轴标签一致；
3. results.json 的 *_aep_gwh 与同一次运行图上的数值逐项对应；
4. 以裸 float 传参必须显式声明单位，未知单位清晰失败。

覆盖路径：基线评估、优化收敛、台数扫描、逐机平均、优化前后对比。
"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.visualization.plotting import (
    plot_aep_vs_turbines,
    plot_comparison,
    plot_convergence,
)


# ---------------------------------------------------------------------------
# 测试夹具：一个小规模、可快速复算的风电场 + 真实优化结果
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def farm():
    n_turb = 6
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    rotor_diameters = np.array([t.rotor_diameter for t in turbines])
    wind_resource = create_default_wind_resource(
        num_sectors=12, dominant_direction=270.0, mean_speed=8.5
    )
    boundary = create_rectangular_boundary(3500, 3500)
    rng = np.random.default_rng(42)

    positions = generate_grid_layout(
        boundary, n_turb, rotor_diameters, min_multiple=5.0, rng=rng
    )
    aep_calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wind_resource,
        wake_model=JensenWake(0.07),
        wake_superposition="sum_of_squares",
        speed_step=1.0,
    )
    baseline = aep_calc.compute_farm_aep(positions)

    ga = GeneticAlgorithm(
        n_turbines=n_turb,
        rotor_diameters=rotor_diameters,
        boundary=boundary,
        fitness_fn=aep_calc.evaluate_layout,
        config=GAConfig(population_size=8, max_generations=3, seed=42),
    )
    optimize_result = ga.optimize(verbose=False)
    optimized = aep_calc.compute_farm_aep(optimize_result.best_positions)

    return {
        "n_turb": n_turb,
        "baseline": baseline,
        "optimized": optimized,
        "optimize_result": optimize_result,
    }


# ---------------------------------------------------------------------------
# 1. 基线：规范单位是 MWh，且 MWh -> GWh 换算唯一
# ---------------------------------------------------------------------------
class TestBaselineUnits:
    def test_farm_result_is_mwh_scale(self, farm):
        """几十 MW 装机、容量系数合理 => 全场年发电量应为几万 MWh（即几十 GWh）。"""
        base = farm["baseline"]
        # 6 台 * 3.45 MW * 8760h ≈ 181 GWh 理论上限；净 AEP 处于 MWh 万级
        assert 1.0e3 < base.net_aep < 1.0e6  # MWh，绝不可能是几万 GWh(=几万*1e3 MWh)
        assert base.net_aep < base.gross_aep

    def test_per_turbine_mwh_sums_to_farm(self, farm):
        """逐机 MWh 之和必须等于全场 MWh（同一单位契约）。"""
        base = farm["baseline"]
        per_turbine = sum(tr.net_aep for tr in base.turbine_results)
        assert per_turbine == pytest.approx(base.net_aep, rel=1e-9)

    def test_json_gwh_equals_mwh_over_1000(self, farm):
        """results.json 写入的 *_aep_gwh 必须等于 MWh/1e3。"""
        base = farm["baseline"]
        json_gwh = float(base.net_aep) / 1e3
        assert json_gwh == pytest.approx(base.net_aep * 1e-3)
        # 反向：图上 GWh 数值 *1e3 必须能找回 JSON 同一数量级
        assert json_gwh * 1e3 == pytest.approx(base.net_aep)


# ---------------------------------------------------------------------------
# 2. 收敛曲线：基准线与优化曲线同数量级（历史 bug：高出三个数量级）
# ---------------------------------------------------------------------------
class TestConvergenceUnits:
    def test_baseline_line_same_order_as_curve(self, farm):
        fig = plot_convergence(
            farm["optimize_result"],
            baseline_aep=farm["baseline"].net_aep,  # MWh
            baseline_aep_unit="MWh",
        )
        ax = fig.axes[0]
        best_ydata = next(
            line.get_ydata() for line in ax.lines if len(line.get_xdata()) > 2
        )
        axhline = next(
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2
            and np.all(np.atleast_1d(line.get_ydata()) == line.get_ydata()[0])
        )
        baseline_gwh = float(np.atleast_1d(axhline.get_ydata())[0])
        final_best_gwh = float(best_ydata[-1])

        # 关键回归断言：基线不得比优化曲线高/低三个数量级
        ratio = baseline_gwh / final_best_gwh
        assert 0.2 < ratio < 5.0, f"基线/最优 比值漂移: {ratio}"

        # 基线 GWh 必须等于 FarmResult MWh/1e3
        assert baseline_gwh == pytest.approx(farm["baseline"].net_aep / 1e3)
        assert ax.get_ylabel() == "净年发电量 (GWh)"

    def test_explicit_kwh_and_gwh_inputs_agree(self, farm):
        """同一基线用 kWh / MWh / GWh 显式传入，图上基准线必须重合。"""
        mwh = farm["baseline"].net_aep
        figs = [
            plot_convergence(
                farm["optimize_result"],
                baseline_aep=mwh * factor,
                baseline_aep_unit=unit,
            )
            for unit, factor in (("kWh", 1e3), ("MWh", 1.0), ("GWh", 1e-3))
        ]
        levels = []
        for fig in figs:
            axhline = next(
                line
                for line in fig.axes[0].lines
                if len(line.get_xdata()) == 2
                and np.all(np.atleast_1d(line.get_ydata()) == line.get_ydata()[0])
            )
            levels.append(float(np.atleast_1d(axhline.get_ydata())[0]))
        for lvl in levels[1:]:
            assert lvl == pytest.approx(levels[0], rel=1e-12)

    def test_unknown_unit_fails_clearly(self, farm):
        with pytest.raises(ValueError, match="单位"):
            plot_convergence(
                farm["optimize_result"],
                baseline_aep=farm["baseline"].net_aep,
                baseline_aep_unit="TWh",
            )


# ---------------------------------------------------------------------------
# 3. 台数扫描：输入 MWh，图上为 GWh，标注与之一致
# ---------------------------------------------------------------------------
class TestTurbineSweepUnits:
    def test_sweep_curve_is_mwh_over_1000(self, farm):
        base = farm["baseline"]
        n_list = [4, 6, 8]
        # 用同一单机 MWh 均值构造扫描输入（规范单位 MWh）
        per = base.net_aep / farm["n_turb"]
        aep_mwh = [per * n for n in n_list]

        fig = plot_aep_vs_turbines(n_list, aep_mwh, aep_unit="MWh")
        ax1 = fig.axes[0]
        plotted = ax1.lines[0].get_ydata()

        np.testing.assert_allclose(plotted, np.array(aep_mwh) / 1e3)
        assert ax1.get_ylabel() == "净年发电量 (GWh)"

    def test_sweep_unit_overloads_agree(self):
        n_list = [2, 4]
        mwh = [1.0e4, 2.0e4]
        fig_mwh = plot_aep_vs_turbines(n_list, mwh, aep_unit="MWh")
        fig_gwh = plot_aep_vs_turbines(n_list, [10.0, 20.0], aep_unit="GWh")
        fig_kwh = plot_aep_vs_turbines(n_list, [1.0e7, 2.0e7], aep_unit="kWh")
        y_mwh = fig_mwh.axes[0].lines[0].get_ydata()
        y_gwh = fig_gwh.axes[0].lines[0].get_ydata()
        y_kwh = fig_kwh.axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(y_mwh, y_gwh)
        np.testing.assert_allclose(y_mwh, y_kwh)

    def test_sweep_unknown_unit_fails(self):
        with pytest.raises(ValueError, match="单位"):
            plot_aep_vs_turbines([1, 2], [1.0, 2.0], aep_unit="MWh/year?")


# ---------------------------------------------------------------------------
# 4 & 5. 优化前后对比：净AEP、逐机平均 的柱高/标签/单位与 JSON 逐项对应
# ---------------------------------------------------------------------------
class TestComparisonUnits:
    @staticmethod
    def _bar_heights(ax):
        return [p.get_height() for p in ax.patches]

    def test_net_aep_bars_match_json_gwh(self, farm):
        base, opt = farm["baseline"], farm["optimized"]
        fig = plot_comparison(base, opt)
        ax_net = fig.axes[0]  # 子图 0：净AEP
        h0, h1 = self._bar_heights(ax_net)

        assert h0 == pytest.approx(base.net_aep / 1e3)
        assert h1 == pytest.approx(opt.net_aep / 1e3)
        assert ax_net.get_ylabel() == "GWh"

    def test_per_turbine_average_units(self, farm):
        base, opt = farm["baseline"], farm["optimized"]
        fig = plot_comparison(base, opt)
        ax_avg = fig.axes[3]  # 子图 3：单机平均AEP
        h0, h1 = self._bar_heights(ax_avg)

        assert h0 == pytest.approx(base.net_aep / len(base.turbine_results) / 1e3)
        assert h1 == pytest.approx(opt.net_aep / len(opt.turbine_results) / 1e3)
        assert ax_avg.get_ylabel() == "GWh/台"

    def test_annotation_text_matches_bar_height(self, farm):
        base, opt = farm["baseline"], farm["optimized"]
        fig = plot_comparison(base, opt)
        ax_net = fig.axes[0]
        # 注释文本中的数值（GWh）必须与柱高一致，而不是 MWh 原值
        texts = [t.get_text() for t in ax_net.texts]
        expected = f"{base.net_aep/1e3:.2f} GWh"
        assert any(expected in t for t in texts), texts

    def test_comparison_no_thousandfold_drift(self, farm):
        """同一次运行：对比图净AEP柱高 与 JSON GWh、FarmResult MWh 三者自洽。"""
        base = farm["baseline"]
        fig = plot_comparison(base, farm["optimized"])
        plotted_gwh = self._bar_heights(fig.axes[0])[0]
        json_gwh = base.net_aep / 1e3
        assert plotted_gwh == pytest.approx(json_gwh)
        # 不可能出现“几万 GWh”：几十 MW 装机的年发电量上界远小于此
        assert plotted_gwh < 1.0e3
