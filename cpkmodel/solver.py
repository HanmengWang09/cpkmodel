"""53个物理状态的前向积分；按心搏、呼吸和输入事件分段。"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .circulation.parameters import finite_real


@dataclass
class SimulationResult:
    times: np.ndarray
    states: np.ndarray
    pressures: np.ndarray
    flows: dict[str, np.ndarray]
    observables: dict[str, np.ndarray]
    statistics: dict


def load_initial_state(path=None):
    """读取普通JSON初态；不读取原项目、实验报告或检查点。"""
    path = Path(path) if path is not None else Path(__file__).parent / 'data' / 'initial_state.json'
    record = json.loads(path.read_text(encoding='utf-8-sig'))
    return np.asarray(record['state'], dtype=float)


def simulate(circuit, protocol, times, initial_state=None, *, method='Radau',
             rtol=1e-8, atol=1e-10, max_step=.01, progress=None):
    """从t=0积分原联立方程，返回完整状态、压力、流量与生理观测量。"""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('输出时间须从0开始、有限且严格递增，至少两个时刻')
    if method not in ('Radau', 'BDF'):
        raise ValueError('求解方法须为Radau或BDF')
    for name, value in [('rtol', rtol), ('atol', atol), ('max_step', max_step)]:
        finite_real(value, name, positive=True)
    x0 = load_initial_state() if initial_state is None else initial_state
    state = circuit.validate_state(x0).copy()
    cuts = circuit.integration_breakpoints(protocol, state, float(times[-1]))
    stops = [0.] + sorted(set(t for t in cuts if 0 < t < times[-1])) + [float(times[-1])]
    states = np.empty((len(times), circuit.state_size))
    statistics = dict(method=method, rtol=rtol, atol=atol, max_step=max_step,
                      nfev=0, njev=0, nlu=0, segments=len(stops) - 1)
    for start, end in zip(stops[:-1], stops[1:]):
        boundary = protocol.at(start)
        solution = solve_ivp(
            lambda t, x: circuit.rates(x, boundary)[0], (start, end), state,
            method=method, jac=lambda t, x: circuit.jacobian(x, boundary),
            rtol=rtol, atol=atol, max_step=max_step, dense_output=True)
        if not solution.success:
            raise RuntimeError(f'{method}积分失败：{solution.message}')
        for accepted in solution.y.T:
            circuit.validate_state(accepted)
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            states[mask] = solution.sol(times[mask]).T
        state = solution.y[:, -1].copy()
        for key in ('nfev', 'njev', 'nlu'):
            statistics[key] += int(getattr(solution, key))
        if progress is not None:
            progress(end, float(times[-1]))
    evaluations = []
    for t, state in zip(times, states):
        circuit.validate_state(state)
        evaluations.append(circuit.evaluate(state, protocol.at(float(t))))
    pressures = np.array([value.pressures for value in evaluations])
    flows = {name: np.array([value.flows[name] for value in evaluations])
             for name in evaluations[0].flows}
    names = ('P_LA', 'P_LV', 'P_RA', 'P_RV', 'P_peri', 'ppl_mmhg',
             'GFR_ml_min', 'RBF_ml_min', 'C_Na_mmol_L', 'Q_urine', 'U_Na',
             'Q_reabs', 'Q_return', 'J_s', 'J_p', 'Q_Ls', 'Q_Lp',
             'q_mouth', 'p_mus', 'ppl', 'V_b', 'W')
    observables = {name: np.array([value.diagnostics[name] for value in evaluations])
                   for name in names if name in evaluations[0].diagnostics}
    return SimulationResult(times.copy(), states, pressures, flows, observables, statistics)
