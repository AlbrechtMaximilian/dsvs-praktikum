import numpy as np
import matplotlib.pyplot as plt


def generate_circle_signal(v, fs, p, snr):
    # Circle parameters
    radius = 1  # Radius of the circle in meters
    T = 2 * np.pi * radius / v  # Time period for one circle
    omega = 2 * np.pi / T  # Angular velocity [rad/s]

    # Total simulation time
    total_time = p * T  # Total time for p periods
    t = np.linspace(0, total_time, int(total_time * fs))  # Time vector

    # Position in 3D space
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    z_amplitude = 0.5  # Amplitude of z oscillation
    z_frequency = 1 / T  # Oscillation frequency for z
    z = z_amplitude * np.sin(2 * np.pi * z_frequency * t)

    # First derivatives (velocity)
    vx = -radius * omega * np.sin(omega * t)
    vy = radius * omega * np.cos(omega * t)
    vz = z_amplitude * 2 * np.pi * z_frequency * np.cos(2 * np.pi * z_frequency * t)

    # Second derivatives (acceleration)
    ax = -radius * omega**2 * np.cos(omega * t)
    ay = -radius * omega**2 * np.sin(omega * t)
    az = -z_amplitude * (2 * np.pi * z_frequency)**2 * np.sin(2 * np.pi * z_frequency * t)

    # Add noise to the signals based on SNR
    def add_noise(signal, snr):
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr / 10))
        noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
        return signal + noise

    ax = add_noise(ax, snr)
    ay = add_noise(ay, snr)
    az = add_noise(az, snr)

    # Store results
    s = np.vstack((x, y, z)).T
    a = np.vstack((t, ax, ay, az)).T

    # Save data
    np.savez("acceleration_circle.npz", acceleration=a, position=s)

    return a, s


def generate_zigzag_signal(A, B, v, fs, p, snr):
    # Compute distance between A and B
    A = np.array(A)
    B = np.array(B)
    distance = np.linalg.norm(B - A)

    # Time to complete one half-cycle (A to B or B to A)
    T_half = distance / (2 * v)  # Half the period
    T = 2 * T_half              # Full period

    # Total simulation time
    total_time = p * T          # Total time for p periods
    t = np.linspace(0, total_time, int(total_time * fs))  # Time vector

    # Direction vector from A to B (normalized)
    direction = (B - A) / distance

    # Generate position signal based on sine function
    position_amplitude = distance / 2
    position = position_amplitude * np.sin((2 * np.pi / T) * t)

    # Velocity (first derivative of position)
    velocity = position_amplitude * (2 * np.pi / T) * np.cos((2 * np.pi / T) * t)

    # Acceleration (second derivative of position)
    acceleration = -position_amplitude * (2 * np.pi / T)**2 * np.sin((2 * np.pi / T) * t)

    # Resolve position, velocity, and acceleration along x, y, z directions
    x = A[0] + direction[0] * (position + position_amplitude)
    y = A[1] + direction[1] * (position + position_amplitude)
    z = A[2] + direction[2] * (position + position_amplitude)

    vx = direction[0] * velocity
    vy = direction[1] * velocity
    vz = direction[2] * velocity

    ax = direction[0] * acceleration
    ay = direction[1] * acceleration
    az = direction[2] * acceleration

    # Add noise to the acceleration signals based on SNR
    def add_noise(signal, snr):
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr / 10))
        noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
        return signal + noise

    ax = add_noise(ax, snr)
    ay = add_noise(ay, snr)
    az = add_noise(az, snr)

    a = np.vstack((t, ax, ay, az)).T
    s = np.vstack((x, y, z)).T

    # Save data
    np.savez("acceleration_zigzag.npz", acceleration=a, position=s)

    return a, s


def plot_acceleration(acceleration):
    t = acceleration[:, 0]
    ax = acceleration[:, 1]
    ay = acceleration[:, 2]
    az = acceleration[:, 3]

    # Plotting acceleration signals
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(t, ax, 'r', linewidth=1.5)
    axes[0].set_title('Acceleration in X-direction')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('a_x (m/s^2)')
    axes[0].grid()

    axes[1].plot(t, ay, 'g', linewidth=1.5)
    axes[1].set_title('Acceleration in Y-direction')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('a_y (m/s^2)')
    axes[1].grid()

    axes[2].plot(t, az, 'b', linewidth=1.5)
    axes[2].set_title('Acceleration in Z-direction')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('a_z (m/s^2)')
    axes[2].grid()

    plt.subplots_adjust(hspace=0.8)  # hspace: vertical distance from subplots
    plt.show()


def plot_trajectory(position):
    # unpack coordinates
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    # Plotting the trajectory in 3D space
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.plot(x, y, z, linewidth=1.5)
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Motion Trajectory')
    plt.grid()
    plt.show()


# Example usage
if __name__ == "__main__":
    # circle signal
    fs = 100  # Sampling frequency [Hz]
    v = 0.1  # Speed [m/s]
    p = 5  # Number of periods
    snr = 30  # Signal-to-noise ratio [dB]
    acceleration_1, position_1 = generate_circle_signal(v, fs, p, snr)


    # zigzag signal
    A = (0, 0, 0)  # Starting point
    B = (1, 1, 1)  # Ending point
    v = 0.5        # Maximum speed [m/s]
    fs = 100       # Sampling frequency [Hz]
    p = 2         # Number of zig-zag periods
    snr = 30       # Signal-to-noise ratio [dB]
    acceleration_2, position_2 = generate_zigzag_signal(A, B, v, fs, p, snr)




