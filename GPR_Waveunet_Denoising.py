import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, BatchNormalization, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load GPR data (Replace with actual data loading code)
def load_gpr_data():
    np.random.seed(42)
    clean_signal = np.sin(np.linspace(0, 20, 100))  # Simulated clean GPR signal
    noise = np.random.normal(0, 0.03, clean_signal.shape)  # Add Gaussian noise
    noisy_signal = clean_signal + noise
    return noisy_signal.reshape(-1, 1), clean_signal.reshape(-1, 1)

# Define Wave-U-Net inspired model for GPR denoising
def build_waveunet(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x1 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(input_layer)
    x1 = BatchNormalization()(x1)
    x2 = Conv1D(128, kernel_size=5, activation='relu', padding='same', strides=2)(x1)
    x2 = BatchNormalization()(x2)
    x3 = Conv1D(256, kernel_size=5, activation='relu', padding='same', strides=2)(x2)
    x3 = BatchNormalization()(x3)
    
    # Decoder with Conv1DTranspose to match shapes
    x4 = Conv1DTranspose(128, kernel_size=5, strides=2, activation='relu', padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Add()([x4, x2])  # Skip connection
    
    x5 = Conv1DTranspose(64, kernel_size=5, strides=2, activation='relu', padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Add()([x5, x1])  # Skip connection
    
    output_layer = Conv1D(1, kernel_size=5, activation='linear', padding='same')(x5)
    
    return Model(input_layer, output_layer)

if __name__ == "__main__":
    # Load Data
    noisy_signal, clean_signal = load_gpr_data()
    noisy_signal = np.expand_dims(noisy_signal, axis=0)  # Add batch dimension
    clean_signal = np.expand_dims(clean_signal, axis=0)
    
    # Build and Compile Model
    model = build_waveunet(input_shape=(100, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    
    # Train Model
    model.fit(noisy_signal, clean_signal, epochs=500, batch_size=1, verbose=1)
    
    # Denoise the signal
    denoised_signal = model.predict(noisy_signal)
    
    # Plot Results
    plt.figure(figsize=(10, 4))
    plt.plot(clean_signal[0], label='Clean Signal', linestyle='dashed')
    plt.plot(noisy_signal[0], label='Noisy Signal', alpha=0.6)
    plt.plot(denoised_signal[0], label='Denoised Signal', linewidth=2)
    plt.legend()
    plt.title("GPR Signal Denoising with Wave-U-Net")
    plt.show()
    
    # Save the trained model
    model.save("gpr_waveunet_denoising.h5")
