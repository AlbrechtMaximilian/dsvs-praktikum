from signal_generator import generate_zigzag_signal, plot_acceleration


acceleration, position = generate_zigzag_signal(A=(0,0,0), B=(1,1,1), v=0.5, fs=100, p=2, snr=30)
plot_acceleration(acceleration)
