import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Multiply, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load & Preprocess Data
df = pd.read_csv('E:\mail\mail_data.csv').fillna('')
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1}).astype(int)
X, Y = df['Message'], df['Category']

# Tokenization
vocab_size = 10000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X_padded, Y, test_size=0.2, random_state=42, stratify=Y)

# Attention Mechanism
def attention_layer(inputs):
    attention_scores = Dense(1, activation='tanh')(inputs)
    attention_scores = Lambda(lambda x: tf.squeeze(x, axis=-1))(attention_scores)
    attention_weights = tf.keras.layers.Softmax()(attention_scores)
    attention_weights = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_weights)
    attended_output = Multiply()([inputs, attention_weights])
    attended_output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_output)
    return attended_output

# Build LSTM Model with Attention
def build_model(vocab_size, embedding_dim, max_length):
    inputs = Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(embedding)
    attention_out = attention_layer(lstm_out)
    output = Dense(1, activation='sigmoid')(attention_out)
    model = Model(inputs, output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train Model (if not already trained)
embedding_dim = 128
model = build_model(vocab_size, embedding_dim, max_length)

try:
    model.load_weights("spam_model.weights.h5")
    print(" Model weights loaded successfully!")
    history = None
except:
    print(" Training the model...")
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)
    model.save_weights("spam_model.weights.h5")
    print(" Model trained and weights saved!")

    # Plotting Training History
    def plot_training_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'r-', label='Training Loss')
        plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    plot_training_history(history)

# Evaluate Model
def evaluate_model(model, X, Y, dataset_name="Dataset"):
    predictions = (model.predict(X) > 0.5).astype(int)
    accuracy = accuracy_score(Y, predictions)
    precision = precision_score(Y, predictions)
    recall = recall_score(Y, predictions)
    f1 = f1_score(Y, predictions)

    print(f"\n {dataset_name} Evaluation Metrics:")
    print(f" Accuracy:  {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1-Score:  {f1:.4f}")

# Evaluate model on training and testing data
evaluate_model(model, X_train, Y_train, "Training Data")
evaluate_model(model, X_test, Y_test, "Testing Data")

# Function to Check If an Email is Spam
def predict_email(email):
    sequence = tokenizer.texts_to_sequences([email])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)[0][0]
    
    if prediction > 0.5:
        print(" The email is NOT SPAM.")
    else:
        print("The email is SPAM!")

# User Input for Email Prediction
while True:
    email_text = input("\nEnter an email to check (or type 'exit' to stop): ")
    if email_text.lower() == "exit":
        print(" Exiting...")
        break
    predict_email(email_text)
