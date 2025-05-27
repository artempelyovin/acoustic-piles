from examples._common import draw_pulse_signal
from utils import _generate_simple_pulse_signal, set_plt_style

if __name__ == "__main__":
    set_plt_style()
    t, pulse, start_x, reflection_x = _generate_simple_pulse_signal(
        fs=1000,
        duration=1.5,
        frequency=6,
        pulse_half_cycles=3,
        pulse_start=0.2,
        pulse_decay=5,
        reflection_delay=0.4,
        reflection_amplitude=0.2,
        reflection_decay=8,
        noise_std=0.0,
    )
    draw_pulse_signal(
        t=t,
        pulse=pulse,
        start_x=start_x,
        reflection_x=reflection_x,
        title="Модель c простейшим затухающим синусом (без шума)",
    )
