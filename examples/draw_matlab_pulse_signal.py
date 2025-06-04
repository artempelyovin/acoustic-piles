from examples._common import draw_pulse_signal
from utils import set_plt_style, load_real_signal_from_6_model


if __name__ == "__main__":
    set_plt_style()
    t, pulse, start_x, reflection_x = load_real_signal_from_6_model()
    draw_pulse_signal(
        t=t,
        pulse=pulse,
        start_x=start_x,
        reflection_x=reflection_x,
        title="Модель реального сгенерированного сигнала в пакете Matlab",
        xlim=(min(t), max(t)),
        ylim=(min(pulse) * 1.2, max(pulse) * 1.2),
    )
