"""读取运行配置、执行前向仿真并导出结果。"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from . import ActiveBoundary, ActiveCircuit, ActiveProtocol, load_reference_parameters
from .solver import load_initial_state, simulate

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, names, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*columns))


def plot_result(out, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for index, label in [(17, '左室'), (19, '右室'), (0, '体动脉'), (12, '肺动脉')]:
        axes[0, 0].plot(result.times, result.pressures[:, index], label=label)
    axes[0, 0].set_ylabel('压力 (mmHg)')
    for index, label in [(31, '左室'), (33, '右室')]:
        axes[0, 1].plot(result.times, result.states[:, index], label=label)
    axes[0, 1].set_ylabel('心腔容积 (mL)')
    for name, label in [('q_AV', '主动脉瓣'), ('q_PVL', '肺动脉瓣')]:
        axes[1, 0].plot(result.times, result.flows[name], label=label)
    axes[1, 0].set_ylabel('流量 (mL/s)')
    axes[1, 1].plot(result.times, result.observables['GFR_ml_min'], label='双肾GFR')
    axes[1, 1].set_ylabel('滤过率 (mL/min)')
    axes[2, 0].plot(result.times, result.states[:, 27], label='肺泡气室')
    axes[2, 0].set_ylabel('气室容积 (L)')
    axes[2, 1].plot(result.times, result.observables['ppl_mmhg'], label='胸膜压')
    axes[2, 1].set_ylabel('压力 (mmHg)')
    for axis in axes.flat:
        axis.grid(alpha=.25)
        axis.legend(loc='upper right', fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('仿真时间 (s)')
    fig.suptitle('心肺肾耦合模型前向仿真')
    fig.savefig(out / 'waveforms.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='53状态心肺肾模型前向仿真')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'default.json', help='运行配置JSON')
    parser.add_argument('--duration', type=float, help='物理仿真时长，秒')
    parser.add_argument('--sample', type=float, help='输出采样间隔，秒')
    parser.add_argument('--method', choices=('Radau', 'BDF'), help='刚性求解器')
    parser.add_argument('--out', type=Path, help='结果目录，默认runs下带时间戳的新目录')
    parser.add_argument('--initial-state', type=Path, help='自定义53状态JSON')
    parser.add_argument('--startup', action='store_true', help='使用构造启动初态替代随附参考初态')
    parser.add_argument('--no-plot', action='store_true', help='仅导出数值结果')
    args = parser.parse_args()
    try:
        if args.startup and args.initial_state:
            raise ValueError('--startup和--initial-state不能同时使用')
        settings = json.loads(args.config.read_text(encoding='utf-8-sig'))
        for key in ('duration', 'sample', 'method'):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        duration = float(settings['duration'])
        sample = float(settings['sample'])
        if not np.isfinite(duration) or not np.isfinite(sample) or duration <= 0 or sample <= 0:
            raise ValueError('时长和采样间隔须为有限正数')
        params = load_reference_parameters()
        boundary = ActiveBoundary(**settings.get('boundary', {}))
        circuit = ActiveCircuit(params)
        state = circuit.reference_initial_state(boundary) if args.startup else load_initial_state(args.initial_state)
        circuit.validate_state(state)
        out = args.out or ROOT / 'runs' / datetime.now().strftime('simulation_%Y%m%d_%H%M%S_%f')
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        times = np.r_[np.arange(0., duration, sample), duration]
        last_printed = -5.
        def progress(current, total):
            nonlocal last_printed
            if current - last_printed >= 5. or current == total:
                print(f'积分进度：{current:.3f} / {total:g} s', flush=True)
                last_printed = current
        metadata = json.loads((Path(__file__).parent / 'data' / 'initial_state.json').read_text(encoding='utf-8'))
        write_json(out / 'configuration.json', dict(settings=settings, parameters=asdict(params),
                   boundary=asdict(boundary), initial_state=state.tolist(), state_names=metadata['state_names'],
                   state_units=metadata['state_units'], initial_state_kind='构造启动初态' if args.startup else 'JSON初态'))
        print(f'开始53状态心肺肾仿真，结果目录：{out}', flush=True)
        started = time.perf_counter()
        result = simulate(circuit, ActiveProtocol((), (boundary,)), times, initial_state=state,
                          method=settings['method'], rtol=settings['rtol'], atol=settings['atol'],
                          max_step=settings['max_step'], progress=progress)
        elapsed = time.perf_counter() - started
        write_csv(out / 'states.csv', ['time_s'] + metadata['state_names'], [times, *result.states.T])
        pressure_names = list(circuit.index) + ['LA', 'LV', 'RA', 'RV']
        write_csv(out / 'pressures.csv', ['time_s'] + [name + '_mmHg' for name in pressure_names],
                  [times, *result.pressures.T])
        write_csv(out / 'flows.csv', ['time_s'] + [name + '_mL_s' for name in result.flows],
                  [times, *result.flows.values()])
        write_csv(out / 'observables.csv', ['time_s'] + list(result.observables), [times, *result.observables.values()])
        np.savez_compressed(out / 'trajectory.npz', time_s=times, states=result.states,
                            pressures_mmHg=result.pressures,
                            **{'flow_' + name: value for name, value in result.flows.items()},
                            **{'observable_' + name: value for name, value in result.observables.items()})
        write_json(out / 'final_state.json', dict(state_names=metadata['state_names'],
                   state_units=metadata['state_units'], state=result.states[-1].tolist()))
        if not args.no_plot:
            plot_result(out, result)
        summary = dict(completed=True, duration_s=duration, sample_count=len(times), state_count=53,
                       integration_wall_seconds=elapsed, solver=result.statistics)
        write_json(out / 'summary.json', summary)
        (out / '运行说明.md').write_text(
            f'# 心肺肾模型前向仿真\n\n已完成{duration:g}秒、{len(times)}个输出时刻的53状态联立积分。'
            f'积分与观测导出准备耗时{elapsed:.2f}秒。\n\n'
            'states.csv为全部状态，pressures.csv为20处压力，flows.csv为通路流量，'
            'observables.csv为生理量；初态、参数和设置见configuration.json。'
            'final_state.json可作为下一段仿真的初态，并保留心搏和呼吸累计相位。\n\n'
            '本报告仅描述前向计算完成，不作稳态、健康成人或临床判断。\n', encoding='utf-8')
        print(f'仿真完成：{duration:g} s，{len(times)}个时刻，53个状态，耗时{elapsed:.2f} s。', flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f'运行失败：{exc}', flush=True)
        return 1
