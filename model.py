import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import ResNet50
import numpy as np
import os
import matplotlib.pyplot as plt

def create_encoder(input_shape=(256, 256, 1), embedding_dim=128):
    """
    Create an encoder model to extract features from patches
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        embedding_dim: Dimension of the output embedding
        
    Returns:
        encoder: Keras Model that outputs embeddings
    """
    # Use appropriate number of channels for input shape
    if input_shape[-1] == 1:
        # For single-channel data, create a custom model
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(embedding_dim)(x)
        x = layers.LayerNormalization()(x)
        encoder = Model(inputs=inputs, outputs=x, name="encoder")
    else:
        # For RGB data, use a pre-trained model
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(embedding_dim)(x)
        x = layers.LayerNormalization()(x)
        encoder = Model(inputs=base_model.input, outputs=x, name="encoder")
    
    return encoder

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function for Siamese network
    
    Args:
        y_true: Ground truth labels (1 for similar, 0 for dissimilar)
        y_pred: Predicted distance between pairs
        margin: Margin for negative pairs
        
    Returns:
        loss: Contrastive loss value
    """
    # Convert to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate loss for similar pairs (y_true==1): d^2
    positive_loss = y_true * tf.square(y_pred)
    
    # Calculate loss for dissimilar pairs (y_true==0): max(0, margin-d)^2
    negative_loss = (1 - y_true) * tf.square(tf.maximum(0., margin - y_pred))
    
    # Return mean loss
    return tf.reduce_mean(positive_loss + negative_loss)

def create_siamese_model(encoder, input_shape=(256, 256, 1)):
    """
    Create a Siamese network for contrastive learning
    
    Args:
        encoder: Encoder model to use for feature extraction
        input_shape: Shape of input images
        
    Returns:
        siamese_model: Model that takes pairs of images and outputs distance
    """
    input_a = layers.Input(shape=input_shape, name="input_a")
    input_b = layers.Input(shape=input_shape, name="input_b")
    
    # Get embeddings for both inputs
    embedding_a = encoder(input_a)
    embedding_b = encoder(input_b)
    
    # Calculate Euclidean distance between embeddings
    distance = layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)),
        name="distance"
    )([embedding_a, embedding_b])
    
    # Create model
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese")
    
    return siamese_model

def train_siamese_model(X1, X2, labels, input_shape=(256, 256, 1), 
                        embedding_dim=128, epochs=20, batch_size=32, 
                        margin=1.0, save_dir="models"):
    """
    Train a Siamese network for contrastive learning
    
    Args:
        X1: First elements of pairs
        X2: Second elements of pairs
        labels: 1 for similar pairs, 0 for dissimilar pairs
        input_shape: Shape of input images
        embedding_dim: Dimension of embeddings
        epochs: Number of training epochs
        batch_size: Batch size for training
        margin: Margin for contrastive loss
        save_dir: Directory to save model and training history
        
    Returns:
        encoder: Trained encoder model
        siamese_model: Trained Siamese model
        history: Training history
    """
    # Create encoder and Siamese model
    encoder = create_encoder(input_shape, embedding_dim)
    siamese_model = create_siamese_model(encoder, input_shape)
    
    # Compile model
    siamese_model.compile(
        loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin),
        optimizer=optimizers.Adam(learning_rate=0.0001)
    )
    
    # Train model
    history = siamese_model.fit(
        [X1, X2], labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model and history
    os.makedirs(save_dir, exist_ok=True)
    encoder.save(os.path.join(save_dir, "encoder.h5"))
    siamese_model.save(os.path.join(save_dir, "siamese.h5"))
    
    # Save training history plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()
    
    return encoder, siamese_model, history

def load_models(save_dir="models"):
    """Load saved encoder and Siamese models"""
    encoder = tf.keras.models.load_model(
        os.path.join(save_dir, "encoder.h5"),
        custom_objects={"contrastive_loss": contrastive_loss}
    )
    siamese_model = tf.keras.models.load_model(
        os.path.join(save_dir, "siamese.h5"),
        custom_objects={"contrastive_loss": contrastive_loss}
    )
    
    return encoder, siamese_model

if __name__ == "__main__":
    # Test model creation
    input_shape = (256, 256, 1)  # Adjust based on your data
    encoder = create_encoder(input_shape)
    siamese_model = create_siamese_model(encoder, input_shape)
    
    # Print model summaries
    print("Encoder Model:")
    encoder.summary()
    
    print("\nSiamese Model:")
    siamese_model.summary()