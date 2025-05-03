from examples._common import draw_pulse_signal
from utils import _generate_complex_pulse_signal, set_plt_style

if __name__ == "__main__":
    set_plt_style()
    t, pulse, start_x, reflection_x = _generate_complex_pulse_signal(
        fs=1000,
        duration=1500.0,
        frequencies=(30, 60, 90, 120),
        decay=5,
        start_time=200.0,
        pulse_duration=200.0,
        reflection_delay=400.0,
        reflection_amp=0.6,
        distortion_level=0.05,
        with_noise=True,
        noise_level=0.25,
    )
    draw_pulse_signal(
        t=t,
        pulse=pulse,
        start_x=start_x,
        reflection_x=reflection_x,
        title="Модель затухающих синусоид (с шумом)",
        ylim=None,
    )
