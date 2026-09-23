"""能量单位契约与换算工具。

全库内部年发电量(AEP)统一使用 **MWh/year**，各边界的单位约定如下：

- ``farm.aep.FarmResult`` / ``TurbineResult`` 的 ``gross_aep``、``net_aep``、
  ``total_wake_loss``、``wake_loss`` 以及 ``sector_results`` 中的
  ``net_aep``/``gross_aep``：MWh/year。
- ``AEPCalculator.evaluate_layout`` 的返回值（优化器适应度）：MWh/year。
- ``optimization.ga.OptimizeResult`` 的 ``best_fitness``、
  ``convergence_history``、``mean_history``：MWh/year。
- 台数扫描在代码内部及 JSON(``aep_mwh``) 中传递的数值：MWh/year。
- 经济性模块 ``EconomicAnalyzer`` 以 GWh 为入参，参数名显式带 ``_GWh`` 后缀。
- 绘图函数只在显示层把 MWh 换算为 GWh；凡接收裸数值的绘图入口，都通过
  ``EnergyUnit`` 显式声明入参单位，不再依赖调用者猜测。
"""

from enum import Enum
from typing import Union

import numpy as np


class EnergyUnit(str, Enum):
    """能量单位（用于年发电量）。"""

    KWH = "kWh"
    MWH = "MWh"
    GWH = "GWh"


#: 库内部统一使用的能量单位
INTERNAL_ENERGY_UNIT = EnergyUnit.MWH

#: 各单位换算到 MWh 的倍数
_TO_MWH: dict[str, float] = {
    EnergyUnit.KWH.value: 1e-3,
    EnergyUnit.MWH.value: 1.0,
    EnergyUnit.GWH.value: 1e3,
}
#: 大小写不敏感的单位名 -> 枚举
_UNIT_ALIASES = {member.value.lower(): member for member in EnergyUnit}

#: MWh -> GWh
MWH_PER_GWH = 1e3

EnergyUnitLike = Union[EnergyUnit, str]


def coerce_unit(unit: EnergyUnitLike) -> EnergyUnit:
    """把枚举或字符串规整为 :class:`EnergyUnit`，无法识别时明确报错。

    Parameters
    ----------
    unit : EnergyUnit | str
        能量单位，接受 ``EnergyUnit`` 或 ``"kWh"``/``"MWh"``/``"GWh"``。
    """
    if isinstance(unit, EnergyUnit):
        return unit
    if isinstance(unit, str):
        resolved = _UNIT_ALIASES.get(unit.strip().lower())
        if resolved is not None:
            return resolved
    raise ValueError(
        f"未知能量单位 {unit!r}，支持: 'kWh'、'MWh'、'GWh'"
        "（或使用 EnergyUnit 枚举）"
    )


def to_mwh(value, unit: EnergyUnitLike):
    """把给定单位的电量值换算为 MWh（标量返回 float，数组形参返回 ndarray）。"""
    factor = _TO_MWH[coerce_unit(unit).value]
    if np.isscalar(value):
        return float(value) * factor
    return np.asarray(value, dtype=np.float64) * factor


def mwh_to(value, unit: EnergyUnitLike):
    """把 MWh 电量值换算为目标单位（标量返回 float，数组形参返回 ndarray）。"""
    factor = _TO_MWH[coerce_unit(unit).value]
    if np.isscalar(value):
        return float(value) / factor
    return np.asarray(value, dtype=np.float64) / factor


def mwh_to_gwh(value):
    """MWh -> GWh 的便捷换算（标量/数组均可）。"""
    return mwh_to(value, EnergyUnit.GWH)
