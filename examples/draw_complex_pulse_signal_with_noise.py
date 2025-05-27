from examples._common import draw_pulse_signal
from utils import _generate_complex_pulse_signal, set_plt_style

if __name__ == "__main__":
    set_plt_style()
    t, pulse, start_x, reflection_x = _generate_complex_pulse_signal(
        fs=1000,
        duration=1.5,
        frequencies=(5, 10),
        pulse_start=0.2,
        pulse_duration=0.09,
        pulse_decay=5,
        reflection_delay=0.3,
        reflection_amplitude=0.17,
        reflection_decay=9,
        noise_std=0.035,
    )
    draw_pulse_signal(
        t=t,
        pulse=pulse,
        start_x=start_x,
        reflection_x=reflection_x,
        title="Модель затухающих синусоид (с шумом)",
        ylim=None,
    )
