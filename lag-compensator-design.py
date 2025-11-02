import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

# --- Data from User's Image (remains constant) ---
log_f_exp = np.array([
    2.0, 2.30103, 2.47712, 2.60206, 2.69897, 2.77815, 2.84510,
    2.90309, 2.95424, 3.0, 3.09691, 3.17609, 3.24304, 3.30103,
    3.35218, 3.39794, 3.43933, 3.47712, 3.51188, 3.54407,
    3.57403, 3.60206, 3.62839, 3.65321, 3.67669, 3.69897
])
log_gain_exp = np.array([
    8.74232, 13.16786, 14.78670, 9.60876, 9.65747, 8.25593, 11.29569,
    12.01875, 11.84131, 15.71656, 20.00868, 21.77689, 21.69153, 22.33215,
    20.65238, 18.36270, 18.04441, 17.23666, 19.74438, 20.57955,
    22.54858, 23.42868, 24.88554, 25.05706, 26.76913, 25.20143
])
phase_diff_exp = np.array([
    115, 52, 22.97, 2.8, 1.84, 2.87, 24.86, 20.14, 35.08, 34.2, 22.69,
    -11.52, -46.6, -63.47, -76.25, -82.94, -98.07, -89.87, -97.77,
    -105.6, -122.4, -138.8, -158, -179, -198.5, -239.1
])

# --- Initial Component Values ---
init_R1 = 3.3e3
init_C1 = 1e-9
init_C = 110e-9
init_R = 1000e3
init_R2 = 51.6e3
init_R3 = 10000e3
init_r = 10e3
init_Q = 5
init_m = 1


fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
plt.subplots_adjust(left=0.1, bottom=0.1)


ax1.plot(log_f_exp, log_gain_exp, marker='o', linestyle='-', color='blue', label='Original Data')
ax2.plot(log_f_exp, phase_diff_exp, marker='o', linestyle='-', color='blue', label='Original Data')


line_theory_gain, = ax1.plot(log_f_exp, np.zeros_like(log_f_exp), marker='x', linestyle='--', color='green', label='Theoretical TF')
line_combined_gain, = ax1.plot(log_f_exp, np.zeros_like(log_f_exp), marker='s', linestyle=':', color='red', label='Combined Response')
line_theory_phase, = ax2.plot(log_f_exp, np.zeros_like(log_f_exp), marker='x', linestyle='--', color='green', label='Theoretical TF')
line_combined_phase, = ax2.plot(log_f_exp, np.zeros_like(log_f_exp), marker='s', linestyle=':', color='red', label='Combined Response')


ax2.axhline(y=-180, color='grey', linestyle=':', linewidth=1.5, label='-180°')

vline_gain = ax1.axvline(x=log_f_exp[0], color='purple', linestyle='-.', label='Crossover Freq')
vline_phase = ax2.axvline(x=log_f_exp[0], color='purple', linestyle='-.')
vline_gain.set_visible(False)
vline_phase.set_visible(False)

gain_margin_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, fontsize=10,
                            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
gain_margin_text.set_visible(False)


ax1.set_title('Bode Plot', fontsize=16)
ax1.set_ylabel('Gain (dB)', fontsize=12)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend()
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.set_xlabel('Log Frequency (log f)', fontsize=12)
ax2.set_ylabel('Phase (degrees)', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend()



def update_plots(R1, C1, C, R, R2, R3, r, Q, m):
    f = 10**log_f_exp
    w = 2 * np.pi * f
    s = 1j * w

    num_s2 = (C1/C)**2
    num_s1 = (1/C) * (1/R1 - r/(R*R3))
    num_s0 = 1 / (C**2 * R * R2)
    numerator = num_s2 * s**2 + num_s1 * s + num_s0
    den_s2 = 1
    den_s1 = 1 / (Q * C * R)
    den_s0 = 1 / (C**2 * R**2)
    denominator = den_s2 * s**2 + den_s1 * s + den_s0
    H_s = m * (numerator / denominator)

    with np.errstate(divide='ignore'):
        mag_db_theory = 20 * np.log10(np.abs(H_s))
    phase_deg_theory = np.angle(H_s, deg=True)
    line_theory_gain.set_ydata(mag_db_theory)
    line_theory_phase.set_ydata(phase_deg_theory)

    mag_linear_exp = 10**(log_gain_exp / 20)
    phase_rad_exp = np.deg2rad(phase_diff_exp)
    H_exp = mag_linear_exp * np.exp(1j * phase_rad_exp)
    H_combined = H_exp * H_s
    with np.errstate(divide='ignore'):
        mag_db_combined = 20 * np.log10(np.abs(H_combined))
    
    phase_rad_combined_unwrapped = np.unwrap(np.angle(H_combined))
    phase_deg_combined = np.rad2deg(phase_rad_combined_unwrapped)

    line_combined_gain.set_ydata(mag_db_combined)
    line_combined_phase.set_ydata(phase_deg_combined)

    all_gain_data = np.concatenate([log_gain_exp, mag_db_theory, mag_db_combined])
    finite_gain_data = all_gain_data[np.isfinite(all_gain_data)]
    if finite_gain_data.size > 0:
        min_gain, max_gain = np.min(finite_gain_data), np.max(finite_gain_data)
        gain_range = max_gain - min_gain if max_gain > min_gain else 1
        ax1.set_ylim(min_gain - 0.1 * gain_range, max_gain + 0.1 * gain_range)

    all_phase_data = np.concatenate([phase_diff_exp, phase_deg_theory, phase_deg_combined])
    finite_phase_data = all_phase_data[np.isfinite(all_phase_data)]
    if finite_phase_data.size > 0:
        min_phase, max_phase = np.min(finite_phase_data), np.max(finite_phase_data)
        phase_range = max_phase - min_phase if max_phase > min_phase else 1
        ax2.set_ylim(min_phase - 0.1 * phase_range, max_phase + 0.1 * phase_range)

    phase = phase_deg_combined
    indices = np.where(np.diff(np.sign(phase - (-180))))[0]
    
    if len(indices) > 0:
        i = indices[0]
        p1, p2 = phase[i], phase[i+1]
        f1, f2 = log_f_exp[i], log_f_exp[i+1]
        
        if p2 - p1 != 0:
            log_f_cross = f1 + (f2 - f1) * (-180 - p1) / (p2 - p1)
            vline_gain.set_xdata([log_f_cross]); vline_phase.set_xdata([log_f_cross])
            vline_gain.set_visible(True); vline_phase.set_visible(True)
            
            gain_at_crossover = np.interp(log_f_cross, log_f_exp, mag_db_combined)
            gain_margin = -gain_at_crossover
            gain_margin_text.set_text(f'Gain Margin: {gain_margin:.2f} dB')
            gain_margin_text.set_visible(True)
        else:
            vline_gain.set_visible(False); vline_phase.set_visible(False)
            gain_margin_text.set_visible(False)
    else:
        vline_gain.set_visible(False); vline_phase.set_visible(False)
        gain_margin_text.set_visible(False)

    fig.canvas.draw_idle()

# --- Slider update function ---
def on_slider_change(val):
    update_plots(R1_slider.val, C1_slider.val, C_slider.val, R_slider.val,
        R2_slider.val, R3_slider.val, r_slider.val, Q_slider.val, m_slider.val)

axcolor = 'lightgoldenrodyellow'
slider_ax_R1 = fig.add_axes([0.15, 0.30, 0.75, 0.02], facecolor=axcolor)
slider_ax_C1 = fig.add_axes([0.15, 0.27, 0.75, 0.02], facecolor=axcolor)
slider_ax_C  = fig.add_axes([0.15, 0.24, 0.75, 0.02], facecolor=axcolor)
slider_ax_R  = fig.add_axes([0.15, 0.21, 0.75, 0.02], facecolor=axcolor)
slider_ax_R2 = fig.add_axes([0.15, 0.18, 0.75, 0.02], facecolor=axcolor)
slider_ax_R3 = fig.add_axes([0.15, 0.15, 0.75, 0.02], facecolor=axcolor)
slider_ax_r  = fig.add_axes([0.15, 0.12, 0.75, 0.02], facecolor=axcolor)
slider_ax_Q  = fig.add_axes([0.15, 0.09, 0.75, 0.02], facecolor=axcolor)
slider_ax_m  = fig.add_axes([0.15, 0.06, 0.75, 0.02], facecolor=axcolor)
reset_ax     = fig.add_axes([0.8, 0.01, 0.1, 0.04])

R1_slider = Slider(slider_ax_R1, 'R1 (kΩ)', 1, 100, valinit=init_R1/1e3, valstep=0.1)
C1_slider = Slider(slider_ax_C1, 'C1 (nF)', 1, 100, valinit=init_C1/1e-9, valstep=0.1)
C_slider  = Slider(slider_ax_C,  'C (nF)',  1, 100, valinit=init_C/1e-9, valstep=0.1)
R_slider  = Slider(slider_ax_R,  'R (kΩ)',  1, 100, valinit=init_R/1e3, valstep=0.1)
R2_slider = Slider(slider_ax_R2, 'R2 (kΩ)', 1, 100, valinit=init_R2/1e3, valstep=0.1)
R3_slider = Slider(slider_ax_R3, 'R3 (kΩ)', 1, 100, valinit=init_R3/1e3, valstep=0.1)
r_slider  = Slider(slider_ax_r,  'r (kΩ)',  0.1, 10, valinit=init_r/1e3, valstep=0.1)
Q_slider  = Slider(slider_ax_Q,  'Q',       0.1, 20, valinit=init_Q, valstep=0.1)
m_slider  = Slider(slider_ax_m,  'm',       0.1, 10, valinit=init_m, valstep=0.1)

sliders = [R1_slider, C1_slider, C_slider, R_slider, R2_slider, R3_slider, r_slider, Q_slider, m_slider]
def update_from_sliders(val):
    update_plots(R1_slider.val * 1e3, C1_slider.val * 1e-9, C_slider.val * 1e-9,
                 R_slider.val * 1e3, R2_slider.val * 1e3, R3_slider.val * 1e3,
                 r_slider.val * 1e3, Q_slider.val, m_slider.val)
    for s in sliders:
        s.valtext.set_text(f'{s.val:.1f}')

for s in sliders:
    s.on_changed(update_from_sliders)
    s.valtext.set_text(f'{s.val:.1f}')

reset_button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    for s in sliders: s.reset()
reset_button.on_clicked(reset)


update_from_sliders(None)

plt.show()