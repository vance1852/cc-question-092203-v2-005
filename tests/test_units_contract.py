"""能量单位契约的回归测试。

验证内部 AEP 数值统一为 MWh/year：
- ``AEPCalculator.evaluate_layout`` 与 ``FarmResult.net_aep`` 同一次布局一致；
- ``OptimizeResult`` 的适应度/历史值与重新评估的 FarmResult 一致；
- 单位换算工具自身正确。
"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.units import EnergyUnit, coerce_unit, mwh_to, mwh_to_gwh, to_mwh


@pytest.fixture(scope="module")
def farm_setup():
    n_turb = 8
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    rotor_diameters = np.array([t.rotor_diameter for t in turbines])
    boundary = create_rectangular_boundary(3500, 3500)
    wind_resource = create_default_wind_resource(
        num_sectors=12, dominant_direction=270.0, mean_speed=8.5
    )
    aep_calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wind_resource,
        wake_model=JensenWake(0.07),
        speed_step=1.0,
    )
    rng = np.random.default_rng(42)
    positions = generate_grid_layout(
        boundary, n_turb, rotor_diameters, min_multiple=5.0, rng=rng
    )
    return aep_calc, boundary, rotor_diameters, positions


def test_unit_conversions():
    assert to_mwh(1.0, EnergyUnit.GWH) == pytest.approx(1000.0)
    assert to_mwh(1000.0, EnergyUnit.KWH) == pytest.approx(1.0)
    assert to_mwh(5.0, "MWh") == pytest.approx(5.0)
    assert mwh_to_gwh(1500.0) == pytest.approx(1.5)
    assert mwh_to(1.0, EnergyUnit.KWH) == pytest.approx(1000.0)

    arr = to_mwh(np.array([1.0, 2.0]), EnergyUnit.GWH)
    np.testing.assert_allclose(arr, [1000.0, 2000.0])

    assert coerce_unit("gwh") is EnergyUnit.GWH
    assert coerce_unit(" MWh ") is EnergyUnit.MWH
    with pytest.raises(ValueError):
        to_mwh(1.0, "TWh")


def test_farm_result_energy_is_mwh(farm_setup):
    aep_calc, _, _, positions = farm_setup
    result = aep_calc.compute_farm_aep(positions)

    # 8 台 3.45MW 风机，满发约 241 GWh/年，净 AEP 应为几万~几十万 MWh，
    # 绝不应是几十（GWh 量级）或几千万（kWh 量级）。
    assert 1e4 < result.net_aep < 3e5
    assert result.gross_aep > result.net_aep > 0
    assert result.total_wake_loss == pytest.approx(
        result.gross_aep - result.net_aep
    )

    # 单机分项之和等于全场值，且同样是 MWh。
    np.testing.assert_allclose(
        sum(tr.net_aep for tr in result.turbine_results),
        result.net_aep,
        rtol=1e-10,
    )
    for sector in result.sector_results.values():
        assert sector["net_aep"] >= 0.0


def test_evaluate_layout_matches_farm_result(farm_setup):
    """优化器适应度必须与 FarmResult.net_aep（MWh）对同一布局一致。"""
    aep_calc, _, _, positions = farm_setup
    fitness_mwh = aep_calc.evaluate_layout(positions)
    result = aep_calc.compute_farm_aep(positions)

    assert fitness_mwh == pytest.approx(result.net_aep, rel=1e-9)
    assert 1e4 < fitness_mwh < 3e5


def test_optimize_result_history_is_mwh(farm_setup):
    aep_calc, boundary, rotor_diameters, positions = farm_setup
    n_turb = len(positions)

    ga = GeneticAlgorithm(
        n_turbines=n_turb,
        rotor_diameters=rotor_diameters,
        boundary=boundary,
        fitness_fn=aep_calc.evaluate_layout,
        config=GAConfig(
            population_size=6,
            max_generations=3,
            min_spacing_multiple=5.0,
            seed=7,
        ),
    )
    opt = ga.optimize(verbose=False)

    assert len(opt.convergence_history) == 3
    assert len(opt.mean_history) == 3

    # best_fitness 与用最优位置重新精算的 FarmResult 净 AEP 一致（MWh）。
    rechecked = aep_calc.compute_farm_aep(opt.best_positions)
    assert opt.best_fitness == pytest.approx(rechecked.net_aep, rel=1e-9)

    # 历史值与 best_fitness 处于同一数量级（MWh），不允许出现 1e3 漂移。
    for value in opt.convergence_history + opt.mean_history:
        assert abs(value) > 1e3 or value < 0  # 惩罚值为负属正常
        assert 0 < value < 3e5 or value < 0
